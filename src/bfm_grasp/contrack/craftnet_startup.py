"""Discard startup dynamics, refresh first-frame contacts, then validate before PPO."""
import json
from pathlib import Path
import torch

@torch.no_grad()
def prepare(env, output):
    from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.state import reset_reference_episode_first_frame
    e=env.unwrapped
    if not hasattr(e,'ref_hands'):return
    devices=[torch.device(e.device).index or 0] if str(e.device).startswith('cuda') else []
    report=dict(enabled=True,controller='zero_residual_reference_pd',num_envs=e.num_envs,
                networks_created=False,training_updates=0,rollouts_collected=0,
                discarded_steps=int(e.ref_len),contact_refresh_steps=2,validation_steps=2,
                speed_threshold_m_s=1.0,reset_protocol='fixed_refresh2_validate2_v2')
    with torch.random.fork_rng(devices=devices):
        env.reset()
        names=['reset_lib_state','reset_lib_track_len','reset_lib_track_len_best','reset_lib_terminate_index_count','reset_lib_reset_index_count','threshold_curriculum_success_ema','object_pos_threshold','object_rot_threshold']
        saved={n:getattr(e,n).clone() for n in names if isinstance(getattr(e,n,None),torch.Tensor)}
        original_counter=e.common_step_counter
        e.episode_valid.zero_();e.episode_filled.zero_()
        reset_reference_episode_first_frame(e,None)
        zero=torch.zeros((e.num_envs,e.action_manager.total_action_dim),device=e.device)
        e._craftnet_warmup=True
        try:
            for i in range(e.ref_len):
                e.episode_valid.zero_()
                env.step(zero)
                if i%100==0:print(f'CRAFTNET_STARTUP step {i}/{e.ref_len}',flush=True)
        finally:
            e._craftnet_warmup=False
        checks=[]
        # A fixed global refresh, never selective retries or filtering bad envs.
        # Every sample starts from the exact original first-frame state.
        for trial in range(4):
            e.episode_valid.zero_();e.episode_filled.zero_()
            env.reset()
            e.episode_valid.zero_();e.episode_filled.zero_()
            reset_reference_episode_first_frame(e,None)
            before=e.scene['objects'].data.object_lin_vel_w.norm(dim=-1)
            e.action_manager.process_action(zero)
            for _ in range(e.cfg.decimation):
                e.action_manager.apply_action();e.scene.write_data_to_sim()
                e.sim.step(render=False);e.scene.update(e.physics_dt)
            speed=e.scene['objects'].data.object_lin_vel_w.norm(dim=-1)
            row=dict(stage='contact_refresh' if trial<2 else 'validation',index=trial,
                     max_speed_m_s=float(speed.max()),speed_gt_1m_s=int((speed>1).sum()),
                     bad_envs=(speed.max(dim=1).values>1).nonzero().flatten().tolist(),
                     finite=bool(torch.isfinite(speed).all()),pre_step_max_speed_m_s=float(before.max()))
            checks.append(row)
            print('CRAFTNET_RESET_CHECK '+json.dumps(row),flush=True)
        valid=checks[2:]
        report.update(checks=checks,probe_max_speed_m_s=max(x['max_speed_m_s'] for x in valid),
                      probe_speed_gt_1m_s=max(x['speed_gt_1m_s'] for x in valid))
        unchanged=all(torch.equal(getattr(e,n),v) for n,v in saved.items() if n not in ('reset_lib_reset_index_count','reset_lib_terminate_index_count'))
        e.episode_valid.zero_();e.episode_filled.zero_()
        env.reset()
        for n,v in saved.items():getattr(e,n).copy_(v)
        e.common_step_counter=original_counter
        report.update(full_reset=True,curriculum_preserved=unchanged,random_state_preserved=True)
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2))
        assert unchanged, 'Startup changed reset curriculum'
        assert all(x['finite'] for x in checks), 'Nonfinite startup state'
        assert all(x['speed_gt_1m_s']==0 for x in valid), 'Post-refresh first-frame impulse remains'
    print('CRAFTNET_STARTUP '+json.dumps(report),flush=True)
