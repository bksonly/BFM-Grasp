"""Discard startup contacts before training or evaluation; never update networks or replay."""
import torch
from motion_contract import tracking_context

@torch.no_grad()
def _prepare(wrapped,env,bridge,agent,refs,groups,window,dims,episodes):
    if not episodes:
        return bridge.observe(),{"enabled":False,"discarded_steps":0}
    seen=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    obs=bridge.observe()
    first_speed=None
    horizon=int(env.group_lengths.max()) if hasattr(env,"group_lengths") else int(env.ref_len)
    for step in range(horizon+1):
        q=env.ref_qpos[env.episode_length_buf.clamp(max=env.ref_len-1),0,7:].clone()
        for group,motion in enumerate(env.hand_motion):
            mask=groups==group
            q[mask]=motion[env.episode_length_buf[mask].clamp(max=len(motion)-1),6:]
        action=(2*(q-bridge.term.lower)/(bridge.term.upper-bridge.term.lower)-1).clamp(-1,1)
        bridge.before_step(obs,action)
        _,_,terminated,truncated,_=wrapped.step(action)
        if step==0:
            first_speed=bridge.reward_state["object_velocity"][:,:3].norm(dim=-1)
        done=terminated|truncated
        seen|=done
        bridge.reset_history(done.nonzero().flatten())
        obs=bridge.observe()
        if bool(seen.all()):break
    if not bool(seen.all()):raise RuntimeError("Warmup did not complete an episode in every environment")
    # Fixed contact refresh then two mandatory probes, all discarded.
    # No retries, no filtering and the 1 m/s threshold is unchanged.
    reset_checks=[]
    ids=torch.arange(env.num_envs,device=env.device)
    for probe in range(4):
        wrapped.reset()
        bridge.reset_history(ids)
        obs=bridge.observe()
        q=torch.empty(env.num_envs,22,device=env.device)
        for group,motion in enumerate(env.hand_motion):q[groups==group]=motion[0,6:]
        action=(2*(q-bridge.term.lower)/(bridge.term.upper-bridge.term.lower)-1).clamp(-1,1)
        bridge.before_step(obs,action)
        wrapped.step(action)
        speed=bridge.reward_state['object_velocity'][:,:3].norm(dim=-1)
        valid=torch.isfinite(speed)&(speed<=1.)
        reset_checks.append(dict(stage='refresh' if probe<2 else 'validation',max_speed_m_s=float(speed.max()),invalid_envs=int((~valid).sum())))
        if probe>=2 and not bool(valid.all()):
            raise RuntimeError(f'Fixed startup reset validation failed: {reset_checks}')
    wrapped.reset()
    ids=torch.arange(env.num_envs,device=env.device)
    bridge.reset_history(ids)
    bridge.captured=None
    if not bool((env.episode_length_buf==0).all()):raise RuntimeError("Startup phase did not reset")
    report={"reset_checks":reset_checks,"reset_protocol":"fixed_refresh2_validate2_v2","controller":"expert_absolute_pd","enabled":True,"discarded_steps":step+1,"completed_envs":int(seen.sum()),"full_reset":True,
            "history_zero":bool((bridge.history.value==0).all()),"last_action_zero":bool((bridge.last==0).all()),
            "warmup_first_step_speed_gt_1m_s_by_group":{str(int(g)):int(((groups==g)&(first_speed>1)).sum()) for g in torch.unique(groups)}}
    return bridge.observe(),report


@torch.no_grad()
def prepare_startup(wrapped,env,bridge,agent,refs,groups,window,dims,episodes):
    model=agent._model
    before={name:value.detach().clone() for name,value in model.named_buffers()}
    was_training=model.training
    devices=[torch.cuda.current_device()] if str(env.device).startswith("cuda") else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            obs,report=_prepare(wrapped,env,bridge,agent,refs,groups,window,dims,episodes)
    finally:
        model.train(was_training)
    unchanged=all(torch.equal(value,before[name]) for name,value in model.named_buffers())
    if not unchanged:raise RuntimeError("Startup changed model/normalizer buffers")
    report["model_buffers_unchanged"]=unchanged
    report["random_state_preserved"]=True
    return obs,report
