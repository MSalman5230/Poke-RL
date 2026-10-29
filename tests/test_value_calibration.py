import numpy as np
from tools.audit_value_calibration import crossfit


def test_value_shrinkage_keeps_seed_pairs_together_and_scores_unseen_games():
    rows=[{'game':game,'outcome':1 if game%2 else -1,'potential':0.,
           'private':3.*(1 if game%2 else -1)} for game in range(12)]
    result=crossfit(rows,'private')
    assert result['raw']['return_mse']==4.
    assert result['out_of_fold_shrinkage']['return_mse']<1e-10
    assert all(np.isclose(f['slope'],1/3) and f['heldout_games']==6 for f in result['fits'])
