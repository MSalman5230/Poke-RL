import pytest
from tools.summarize_validation_families import group_results


def test_family_diagnostics_follow_learner_seat_and_keep_truncated_games():
    games=[('balanced',{'game':0,'seat':0,'won':True,'truncated':False,'team_families':['rain','sand']}),
           ('balanced',{'game':1,'seat':1,'won':False,'truncated':True,'team_families':['rain','sand']})]
    result=group_results(games)
    assert result['own_family']['rain']['wins']==1
    assert result['own_family']['sand']['wins']==0
    assert result['enemy_family']['sand']['wins']==1
    assert result['opponent']['balanced']=={'games':2,'wins':1,'win_rate':.5,'seed_blocks':1,'truncations':1}
    with pytest.raises(ValueError,match='failed validation'):
        group_results([('balanced',{'error':'engine failure'})])
