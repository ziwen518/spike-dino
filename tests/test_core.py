import torch
from spike_dino.core import DinoLoss, Experiment, independent_forward

torch.set_num_threads(2)


def test_dino_excludes_matching_views_and_detaches_teacher():
    loss_fn = DinoLoss(3, student_temp=1., teacher_temp=1.)
    s = [torch.randn(2, 2, 3, requires_grad=True) for _ in range(3)]
    t = [torch.randn(2, 2, 3, requires_grad=True) for _ in range(2)]
    p = [z.mean(0).softmax(-1) for z in t]
    q = [z.mean(0).log_softmax(-1) for z in s]
    expected = sum(-(p[i]*q[j]).sum(-1).mean() for i,j in [(0,1),(0,2),(1,0),(1,2)])/4
    actual = loss_fn(s,t)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(z.grad is None for z in t)
    assert all(z.grad is not None for z in s)


def test_state_isolation_and_teacher_update():
    torch.manual_seed(1)
    model = Experiment(width=.0625, hidden=32, bottleneck=8, prototypes=16, teacher_momentum=.5)
    model.eval()
    x = torch.rand(2,2,2,16,16)
    with torch.no_grad():
        first = independent_forward(model.student, x)[1]
        independent_forward(model.student, x*7)
        again = independent_forward(model.student, x)[1]
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    model.train()
    previous = [p.clone() for p in model.teacher.parameters()]
    loss, _, logits = model([x,x.flip(-1)])
    loss.backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.backbone.parameters())
    opt = torch.optim.SGD(model.student.parameters(), lr=.01)
    opt.step()
    center_expected = torch.cat([z.detach().mean(0) for z in logits]).mean(0,keepdim=True)*.1
    model.after_optimizer_step(logits)
    for old,t,s in zip(previous,model.teacher.parameters(),model.student.parameters()):
        torch.testing.assert_close(t,(old+s)/2)
    torch.testing.assert_close(model.criterion.center,center_expected)
    for t,s in zip(model.teacher.buffers(),model.student.buffers()):
        torch.testing.assert_close(t,s)


def test_contrastive_losses_match_manual_reference():
    from third_party.eventclr.ntxent import TemporalNTXentLoss
    import torch.nn.functional as F
    a,b = torch.randn(3,4,7),torch.randn(3,4,7)
    def reference(x,y):
        z=F.normalize(torch.cat([x,y]),dim=-1)
        scores=z@z.T/.2
        scores.fill_diagonal_(-torch.inf)
        return F.cross_entropy(scores,(torch.arange(8)+4)%8)
    torch.testing.assert_close(TemporalNTXentLoss(.2,'naive')(a,b),reference(a.mean(0),b.mean(0)))
    torch.testing.assert_close(TemporalNTXentLoss(.2,'temporal')(a,b),torch.stack([reference(x,y) for x,y in zip(a,b)]).mean())


def test_temporal_bn_matches_pooled_batch_statistics():
    from spike_dino.core import TemporalBatchNorm1d
    actual = TemporalBatchNorm1d(5, step_mode='m')
    reference = torch.nn.BatchNorm1d(5)
    reference.load_state_dict(actual.state_dict())
    x = torch.randn(3,4,5)
    torch.testing.assert_close(actual(x), reference(x.flatten(0,1)).reshape_as(x))
    torch.testing.assert_close(actual.running_var,reference.running_var)


def test_checkpoint_restores_teacher_center_and_student(tmp_path):
    model = Experiment(width=.0625, hidden=32, bottleneck=8, prototypes=16)
    model.criterion.center.fill_(.125)
    path = tmp_path/'state.pt'
    torch.save(model.state_dict(),path)
    restored = Experiment(width=.0625, hidden=32, bottleneck=8, prototypes=16)
    restored.load_state_dict(torch.load(path,weights_only=True))
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[name])


def test_strict_spike_dino_binary_communication_and_gradients():
    torch.manual_seed(3)
    model = Experiment(method='spike_dino', width=.125, hidden=64, prototypes=16)
    x = torch.rand(4, 4, 2, 32, 32)
    x *= (torch.rand_like(x) < .2)
    residual_values = []
    def check_residual(_module, _inputs, output):
        residual_values.append(output.detach())
    from third_party.eventclr.models.sew_resnet import SEWBasicBlock
    hooks = [block.register_forward_hook(check_residual) for block in model.student.backbone.modules()
             if isinstance(block, SEWBasicBlock)]
    try:
        features, prototype_spikes = independent_forward(model.student, x)
    finally:
        for hook in hooks:
            hook.remove()
    for value in [*residual_values, features, prototype_spikes]:
        assert bool(torch.all((value == 0) | (value == 1)))
    assert len(residual_values) == 8
    assert 0 < model.student.last_rates['input_spike_rate'] < 1
    assert model.student.last_rates['backbone_output_rate'] > 0
    assert model.student.last_rates['head_prototype_rate'] > 0
    loss, _, teacher_spikes = model([x, x.flip(-1)])
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.all((value == 0) | (value == 1))) for value in teacher_spikes)
    loss.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in model.student.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_spike_prototype_threshold_controller_and_checkpoint(tmp_path):
    model = Experiment(method='spike_dino', width=.0625, hidden=32,
                       prototypes=16, output_target_rate=.05)
    head = model.student.head
    original = head.prototype_threshold.item()
    head.last_rates = {'head_prototype_rate': .01}
    head.adjust_output_threshold(.05)
    assert head.prototype_threshold.item() < original
    head.last_rates = {'head_prototype_rate': .2}
    adjusted = head.prototype_threshold.item()
    head.adjust_output_threshold(.05)
    assert head.prototype_threshold.item() > adjusted
    path = tmp_path / 'strict_state.pt'
    torch.save(model.state_dict(), path)
    restored = Experiment(method='spike_dino', width=.0625, hidden=32,
                          prototypes=16, output_target_rate=.05)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.student.head.prototype_threshold,
                               head.prototype_threshold)


def test_bernoulli_input_coding_is_sparse_and_binary():
    torch.manual_seed(7)
    from spike_dino.core import StrictSpikeEncoder
    encoder = StrictSpikeEncoder(width=.125, hidden=64, prototypes=16,
                                 input_coding='bernoulli')
    seen = []
    hook = encoder.backbone.conv1.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach()))
    x = torch.full((4, 4, 2, 32, 32), .1)
    try:
        h, z = independent_forward(encoder, x)
    finally:
        hook.remove()
    assert len(seen) == 1
    assert bool(torch.all((seen[0] == 0) | (seen[0] == 1)))
    assert .07 < seen[0].float().mean().item() < .13
    assert bool(torch.all((h == 0) | (h == 1)))
    assert bool(torch.all((z == 0) | (z == 1)))


def test_population_spikes_decode_only_at_dino_loss():
    torch.manual_seed(17)
    model = Experiment(method='spike_dino', width=.125, hidden=64,
                       prototypes=16, population=4, input_coding='bernoulli')
    x = torch.rand(4, 4, 2, 32, 32)
    features, raw_spikes = independent_forward(model.student, x)
    assert raw_spikes.shape == (4, 4, 64)
    assert bool(torch.all((raw_spikes == 0) | (raw_spikes == 1)))
    decoded = model.criterion.rate_logits(raw_spikes)
    assert decoded.shape == (4, 16)
    assert bool(torch.all((decoded >= 0) & (decoded <= 1)))
    loss, _, teacher = model([x, x.flip(-1)])
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in model.student.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_teacher_occupancy_and_student_bernoulli_are_both_spike_inputs():
    torch.manual_seed(9)
    model = Experiment(method='spike_dino', width=.125, hidden=64,
                       prototypes=16, input_coding='bernoulli',
                       teacher_input_coding='occupancy')
    assert model.student.input_coding == 'bernoulli'
    assert model.teacher.input_coding == 'occupancy'
    observed = {'student':[], 'teacher':[]}
    hooks = []
    for name in observed:
        first_conv = getattr(model,name).backbone.conv1
        hooks.append(first_conv.register_forward_pre_hook(
            lambda module,args,key=name:observed[key].append(args[0].detach())))
    x = torch.full((4,4,2,32,32),.2)
    try:
        loss, _, teacher_logits = model([x,x])
    finally:
        for hook in hooks:hook.remove()
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.all((value == 0) | (value == 1)))
               for values in observed.values() for value in values)
    assert all(value.float().mean() == 1 for value in observed['teacher'])
    assert all(.15 < value.float().mean() < .25 for value in observed['student'])
    assert all(bool(torch.all((value == 0) | (value == 1))) for value in teacher_logits)
