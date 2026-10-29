import numpy as np
from champions_rl.environment import ChampionsEnv


def test_gym_contract_and_seeded_legal_episode():
    with ChampionsEnv(opponent="random",split="validation",max_turns=30) as env:
        obs,info=env.reset(seed=79)
        first={k:v.copy() for k,v in obs.items()}
        again,_=env.reset(seed=79)
        assert all(np.array_equal(first[k],again[k]) for k in first)
        assert env.observation_space.contains(obs)
        for _ in range(500):
            action=int(np.flatnonzero(info["action_mask"])[0])
            obs,reward,terminated,truncated,info=env.step(action)
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            if terminated or truncated:
                assert info["trace"]
                break
        else:
            raise AssertionError("Episode failed to finish")
