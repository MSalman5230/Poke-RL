from champions_rl.evaluation import evaluate


def test_saved_roster_test_uses_new_battle_seeds_and_preserves_explicit_seed(tmp_path,monkeypatch):
    observed=[]
    def match(engine,teams,bots,seed,progress):
        observed.append(seed)
        return {'winner':0,'truncated':False,'trace':[]}
    monkeypatch.setattr('champions_rl.evaluation.play_match',match)
    common=dict(opponents=['balanced'],games=2,roster='saved',learner_style='balanced')
    validation=evaluate(output=tmp_path/'validation',split='validation',**common)
    test=evaluate(output=tmp_path/'test',split='test',**common)
    explicit=evaluate(output=tmp_path/'explicit',split='test',seed=71,**common)
    assert observed[:2]==[validation['seed']]*2
    assert observed[2:4]==[test['seed']]*2
    assert not set(observed[:2]) & set(observed[2:4])
    assert observed[4:]==[71,71] and explicit['seed']==71
