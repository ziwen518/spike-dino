from spike_dino.downstream import cohorts


def test_fewshot_cohorts_are_exact_disjoint_and_repeatable():
    split={'train':[{'path':f'{label}/{index}.aedat','label':label}
                    for label in range(10) for index in range(20)]}
    first_train, first_validation=cohorts(split,'3',42,validation_per_class=4)
    second_train, second_validation=cohorts(split,'3',42,validation_per_class=4)
    assert first_train==second_train and first_validation==second_validation
    assert len(first_train)==30 and len(first_validation)==40
    assert {r['path'] for r in first_train}.isdisjoint({r['path'] for r in first_validation})
    assert all(sum(r['label']==label for r in first_train)==3 for label in range(10))
