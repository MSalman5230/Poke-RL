import numpy as np
from tools.audit_value_calibration import crossfit,paired_head_comparison


def test_value_shrinkage_keeps_seed_pairs_together_and_scores_unseen_games():
    rows=[{'game':game,'outcome':1 if game%2 else -1,'potential':0.,
           'private':3.*(1 if game%2 else -1)} for game in range(12)]
    result=crossfit(rows,'private')
    assert result['raw']['return_mse']==4.
    assert result['out_of_fold_shrinkage']['return_mse']<1e-10
    assert all(np.isclose(f['slope'],1/3) and f['heldout_games']==6 for f in result['fits'])


def test_head_comparison_uses_correlated_seed_blocks_and_handles_identical_heads():
    rows=[{'game':game,'outcome':1 if game%2 else -1,'potential':0.,
           'public':0.,'private':(.5+game/30)*(1 if game%2 else -1)} for game in range(12)]
    result=paired_head_comparison(rows,repeats=500)
    assert result['seed_blocks']==6
    assert result['private_minus_public_ev']['bootstrap_95'][0]>0
    # Repeating correlated observations must not manufacture extra evidence.
    duplicated=paired_head_comparison([row for row in rows for _ in range(3)],repeats=500)
    assert np.allclose(result['private_minus_public_ev']['bootstrap_95'],duplicated['private_minus_public_ev']['bootstrap_95'])
    identical=paired_head_comparison([{**row,'private':row['public']} for row in rows],repeats=500)
    assert identical['private_minus_public_ev']['estimate']==0
    assert identical['private_minus_public_ev']['bootstrap_95']==[0.,0.]
