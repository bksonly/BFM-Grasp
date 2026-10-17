"""Single-scene online BFM-Zero bring-up. Multi-scene training is not implied."""
import argparse,os,sys,json,time,itertools,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).resolve().parents[3]/'third_party/BFM-Zero'),str(Path(__file__).resolve().parents[3]/'third_party/ConTrack/source/ConTrack')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--reference-source',choices=['measured','graspxl-geometric'],default='measured')
p.add_argument('--reference',required=True,help='Original reference without lift, already at control FPS')
p.add_argument('--motion',required=True,help='External wrist trajectory including appended lift')
p.add_argument('--expert',action='append',default=[],help='Additional original reference H5s, same robot')
p.add_argument('--group-motion',action='append',default=[])
p.add_argument('--hand-only',action='store_true')
p.add_argument('--reference-replay',action='store_true')
p.add_argument('--kinematic-video',action='store_true',help='Render original hand AND object states without physics integration')
p.add_argument('--updates-per-step',type=int,default=1)
p.add_argument('--replay-capacity',type=int,default=262144)
p.add_argument('--replay-device',choices=['cpu','cuda'],default='cpu')
p.add_argument('--output',required=True)
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--steps',type=int,default=128)
p.add_argument('--min-train-seconds',type=float,default=0)
p.add_argument('--tb-interval',type=int,default=128)
p.add_argument('--warmup',type=int,default=32)
p.add_argument('--update-every',type=int,default=4)
p.add_argument('--batch',type=int,default=32)
p.add_argument('--goal-ratio',type=float,default=.2,choices=[0.,.2],help='Replay-goal fraction; expert fixed at .6, remainder random')
p.add_argument('--network-width',type=int,default=128)
p.add_argument('--reward-mode',choices=['conditional','rp-only'],default='conditional')
p.add_argument('--resume')
p.add_argument('--replay-bank',help='Directory of per-object physical transition shards')
p.add_argument('--object-key',default='box')
p.add_argument('--z-window',type=int,default=8,choices=[1,8],help='Upstream tracking: average future B embeddings, shorten window at trajectory end; 1 is diagnostic only')
p.add_argument('--collect-only',action='store_true')
p.add_argument('--tactile-validation',action='store_true',help='Small diagnostic only: reference PD, never a policy training result')
p.add_argument('--tactile-probe',action='store_true',help='Diagnostic only: synthetic contact sphere per environment')
p.add_argument('--evaluate',action='store_true')
p.add_argument('--startup-warmup-episodes','--eval-warmup-episodes',dest='startup_warmup_episodes',type=int,default=1,choices=[0,1],help='Discard startup episode before training/evaluation, then fully reset; 0 for diagnostics only')
p.add_argument('--video',action='store_true')
p.add_argument('--camera-env',type=int,default=0)
p.add_argument('--save-every',type=int,default=1000)
p.add_argument('--save-seconds',type=float,default=3600)
p.add_argument('--profile-phases',action='store_true')
p.add_argument('--tactile',choices=['none','binary','magnitude','vector'],default='binary')
p.add_argument('--seed',type=int,default=42)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
assert args.evaluate and args.hand_only and args.num_envs==10, 'This entry is evaluation-only, ten objects on one table'
if args.tactile_validation:
    assert args.num_envs<=32 and args.collect_only and not args.evaluate
if args.tactile_probe:raise ValueError('Old five-elastomer probe is incompatible with ten-body object-filtered contacts')
if args.reference_replay:assert args.evaluate
if args.kinematic_video:assert args.hand_only and args.video and args.num_envs==1
os.environ['CONTRACK_RBY1A_SHARPA_DATA']=args.motion
if args.video:args.enable_cameras=True
app=AppLauncher(args).app
import gymnasium as gym
import numpy as np
import h5py
import torch
torch.set_float32_matmul_precision('high')
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg,ObservationTermCfg,TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
import ConTrack.tasks
from core import make_agent,ExpertSampler,Replay,BalancedReplay
from expert_sampler_vectorized import VectorizedExpertSampler
from reference import load_reference
from motion_contract import grouped_reference_mask,tracking_context
from isaac_bridge import HandAction,HandActionCfg,Bridge,capture_terminal,unused_observation
if args.hand_only:
    from hand_only import HandOnlyAction
    HandAction=HandOnlyAction

@configclass
class Actions:
    hand=HandActionCfg(class_type=HandAction,asset_name='rby1a_sharpa',reference_path=args.motion)
@configclass
class Obs:
    @configclass
    class Policy(ObservationGroupCfg):
        placeholder=ObservationTermCfg(func=unused_observation)
        concatenate_terms=True
    policy=Policy()
@configclass
class Rewards:pass
@configclass
class Terminations:
    horizon=TerminationTermCfg(func=capture_terminal,time_out=False)

torch.manual_seed(args.seed)
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
cfg=parse_env_cfg('Isaac-Rby1a-Sharpa-Mimic-Manager-Play-v0',device=args.device,num_envs=args.num_envs)
cfg.seed=args.seed;cfg.actions=Actions();cfg.observations=Obs();cfg.rewards=Rewards();cfg.terminations=Terminations()
cfg.events.advance_frame=None;cfg.events.object_xy_rand=None;cfg.events.joint_pd_rand=None
cfg.threshold_curriculum_enabled=False
from ten_body_contacts import configure_tactile
configure_tactile(cfg)
if args.tactile_probe:
    from tactile_sensors import configure_probe,place_probe
    configure_probe(cfg)
assert len(args.group_motion)==len(args.expert), 'One reference per physical object required'
if args.hand_only:
    from hand_only import configure
    configure(cfg,[args.motion,*args.group_motion])
elif args.group_motion:
    from grouped_scene import configure
    assert len(args.group_motion)==len(args.expert)
    configure(cfg,[args.motion,*args.group_motion])
cfg.sim.physx.gpu_max_rigid_patch_count=2097152
if args.video:
    with h5py.File(args.motion) as f:center=f['object_tracks/0/translations'][0]
    cfg.viewer.origin_type='env';cfg.viewer.env_index=args.camera_env
    cfg.viewer.lookat=tuple(float(x) for x in center)
    cfg.viewer.eye=tuple(float(x+y) for x,y in zip(center,(.45,.45,.3)))
import shared_table
shared_table.configure(cfg,[args.motion,*args.group_motion])
wrapped=gym.make('Isaac-Rby1a-Sharpa-Mimic-Manager-Play-v0',cfg=cfg,render_mode='rgb_array' if args.video else None)

env=wrapped.unwrapped
shared_table.place(env,args.output)
wrapped.reset()
if os.environ.get('CRAFTNET_COLLISION_AUDIT')=='1':
    from collision_audit import check
    check(out)
if args.kinematic_video:
    from kinematic_video import record
    record(env,args.reference,out)
    wrapped.close();app.close();sys.exit(0)
height_audit=None
if os.environ.get('CRAFTNET_HEIGHT_AUDIT')=='1':
    from height_audit import HeightAudit
    height_audit=HeightAudit(env,args.motion,out)
with h5py.File(args.motion) as f:vertices=torch.tensor(f['object_tracks/0/vertices'][:],dtype=torch.float32,device=env.device)
bridge=Bridge(env,vertices,args.tactile);env.bfm_bridge=bridge
agent,dims=make_agent(args.tactile,'cpu',args.network_width,args.batch,goal_ratio=args.goal_ratio)
agent.reward_mode=args.reward_mode
update_offset=0
if args.resume:
    saved=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
    assert saved['dimensions']==dims,'Resume observation contract mismatch'
    assert saved.get('contract_version')=='official_bfmzero_hand_v3','Legacy checkpoint semantics mismatch'
    assert saved.get('reward_mode')==args.reward_mode,'Resume reward mode mismatch'
    agent._model.load_state_dict(saved['model'])
    # Evaluation needs model and normalization buffers only; optimizer state stays on CPU.
    update_offset=saved.get('total_updates',0)
    del saved
    if str(env.device).startswith('cuda'):torch.cuda.empty_cache()
# Only inference modules need GPU residency during this evaluation-only entry.
for module in [agent._model._actor,agent._model._backward_map,agent._model._obs_normalizer]:
    module.to(env.device)
agent._model.device=torch.device(env.device).type
refs=[load_reference(x,env.device,source=args.reference_source) for x in [args.reference,*args.expert]]
expert=VectorizedExpertSampler(refs,dims,env.device)
replay_device=env.device if args.replay_device=='cuda' else 'cpu'
replay=Replay(args.replay_capacity,replay_device)
group_replays=[replay]
if args.group_motion:group_replays += [Replay(args.replay_capacity,replay_device) for _ in args.group_motion]
retained=[]
if args.replay_bank:
    bank=Path(args.replay_bank);bank.mkdir(parents=True,exist_ok=True)
    for shard in sorted(bank.glob('*.pt')):
        if shard.stem==args.object_key:continue
        data=torch.load(shard,map_location=env.device,weights_only=False)
        assert data['dimensions']==dims
        cache=Replay(32768,env.device);cache.add(data['transitions']);retained.append(cache)
training_replay=BalancedReplay(replay,[*group_replays[1:],*retained],block_size=8)
obs=bridge.observe();updates=0;episodes=0;metrics={}
expected=refs[0].corners[0].expand(args.num_envs,-1).clone()
if args.group_motion:
    for group,ref in enumerate(refs):expected[env.group_id==group]=ref.corners[0]
corner_error=float((obs['corners']-expected).abs().max())
assert corner_error<.003,f'Live/reference wrist coordinate mismatch: {corner_error}'
random_z=agent._model.sample_z(args.num_envs,device=env.device)
physical_groups=env.group_id if hasattr(env,'group_id') else torch.zeros(args.num_envs,dtype=torch.long,device=env.device)
ref_mask=grouped_reference_mask(physical_groups)
(out/'exploration-groups.json').write_text(json.dumps([
    {'group':int(group),'envs':int((physical_groups==group).sum()),
     'reference_z_envs':int(ref_mask[physical_groups==group].sum()),
     'random_z_envs':int((~ref_mask[physical_groups==group]).sum())}
    for group in torch.unique(physical_groups)],indent=2))
initial_z=env.scene['objects'].data.object_pos_w[:,0,2].clone()
with h5py.File(args.motion) as f:table_z=float(f.attrs['table_top_z'])
seen=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
hold_min=torch.full((args.num_envs,),float('inf'),device=env.device)
hold_bottom=hold_min.clone();evaluation=[]
writer=None;success_log=None
if not args.evaluate:
    from torch.utils.tensorboard import SummaryWriter
    from training_log import EpisodeSuccessLog
    writer=SummaryWriter(str(out/'tensorboard'),flush_secs=15)
    group_ids=env.group_id if args.group_motion else torch.zeros(args.num_envs,dtype=torch.long,device=env.device)
    success_log=EpisodeSuccessLog(writer,group_ids,ref_mask,group_names={i:Path(m).parent.name for i,m in enumerate([args.motion,*(args.group_motion or [])])})
    writer.add_text('configuration',json.dumps(vars(args),indent=2),0)
    writer.add_text('success_definition','Last 60 control steps: minimum COM rise >= 0.04 m AND minimum mesh bottom clearance > 0.01 m. At least 60 samples required. Rolling last 4096 episodes per series. Training includes exploration; not deterministic evaluation.',0)
contract=dict(actor_keys=['proprio','tactile','last_action','history_actor'],B_keys=['proprio','corners','contact_binary'],F_privileged=['corners','object_velocity'],
    action='22 fixed joint-limit-normalized absolute PD targets; no hand reference residual',B_tail='upstream shortened future window, then hold final z; successful reference retains complete lift/hold',
    scene_scope='simultaneous grouped physical objects' if args.group_motion else 'single physical object',tactile=args.tactile,
    terminal_next_state='captured after physics and before auto-reset',algorithm='upstream FBcprAuxAgent losses with conditional ConTrack auxiliary reward adapter',
    reward='rp for all; expert-source z adds original hand/object/contact terms averaged across its encoded 8-state window; dt=1/60; no ConTrack CMDP scheduler inside BFM',
    expert_target_matching='expert z and reward target window permuted together within the physical object group',
    runtime=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                 total_replay_capacity=args.replay_capacity*len(group_replays),replay_device=str(replay_device),expert_sampler='vectorized'),args=vars(args))
(out/'contract.json').write_text(json.dumps(contract,indent=2));(out/'agent_config.json').write_text(agent.cfg.model_dump_json(indent=2))
if args.group_motion:
    (out/'groups.json').write_text(json.dumps({'counts':torch.bincount(env.group_id).tolist(),'motions':[args.motion,*args.group_motion],'initial_corner_error_m':corner_error},indent=2))
from best_success import improves
best_record=None
if not args.evaluate and (out/'best.ckpt').exists():
    best_saved=torch.load(out/'best.ckpt',map_location='cpu',weights_only=False,mmap=True)
    best_record=best_saved['best_success']
    del best_saved

def save_checkpoint(name,step):
    optimizers={k:v.state_dict() for k,v in vars(agent).items() if isinstance(v,torch.optim.Optimizer)}
    destination=out/name
    temporary=destination.with_suffix('.tmp')
    torch.save({'model':agent._model.state_dict(),'optimizers':optimizers,'step':step,'total_updates':update_offset+updates,'seed':args.seed,'dimensions':dims,
                'contract_version':'official_bfmzero_hand_v3','goal_ratio':args.goal_ratio,'reward_mode':args.reward_mode,'best_success':best_record},temporary)
    temporary.replace(destination)
stop_requested=False
def request_stop(signum,frame):
    global stop_requested
    stop_requested=True
signal.signal(signal.SIGTERM,request_stop)
signal.signal(signal.SIGINT,request_stop)
def begin_phase():
    torch.cuda.synchronize(env.device)
    torch.cuda.reset_peak_memory_stats(env.device)
    return time.monotonic()
def end_phase(name,step,began):
    torch.cuda.synchronize(env.device)
    free,total=torch.cuda.mem_get_info(env.device)
    record=dict(phase=name,step=step,seconds=time.monotonic()-began,
                allocated_gib=torch.cuda.memory_allocated(env.device)/1024**3,
                peak_allocated_gib=torch.cuda.max_memory_allocated(env.device)/1024**3,
                reserved_gib=torch.cuda.memory_reserved(env.device)/1024**3,
                device_used_gib=(total-free)/1024**3)
    with (out/'phase-profile.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
try:
    from startup_warmup import prepare_startup
    obs,warmup_report=prepare_startup(wrapped,env,bridge,agent,refs if args.group_motion else refs[:1],physical_groups,args.z_window,dims,args.startup_warmup_episodes)
    initial_z=env.scene['objects'].data.object_pos_w[:,0,2].clone()
    assert all(buffer.size==0 for buffer in group_replays), 'Startup samples entered replay'
    warmup_report['replay_entries']=sum(buffer.size for buffer in group_replays)
    warmup_report['training_updates']=updates
    (out/'startup-warmup.json').write_text(json.dumps(warmup_report,indent=2))
    if args.evaluate:(out/'evaluation-warmup.json').write_text(json.dumps(warmup_report,indent=2))
    if args.video:
        # Prime the RGB annotator before recording, without advancing physics.
        for _ in range(30):
            env.sim.render();frame=wrapped.render()
            if frame is not None and frame.max()>20:break
        assert frame is not None and frame.max()>20, 'Renderer returned black frames'
        wrapped=gym.wrappers.RecordVideo(wrapped,video_folder=str(out/'videos'),step_trigger=lambda t:t==0,video_length=args.steps)
    started=time.monotonic()
    last_saved=started
    for step in itertools.count():
        if args.profile_phases:phase_started=begin_phase()
        with torch.no_grad():
            refresh=(env.episode_length_buf%agent.cfg.train.update_z_every_step==0)[:,None]
            if agent.cfg.train.use_mix_rollout and not agent.z_buffer.empty():
                candidate=agent.z_buffer.sample(args.num_envs,device=env.device)
            else:candidate=agent._model.sample_z(args.num_envs,device=env.device)
            random_z=torch.where(refresh,candidate,random_z)
            zref=torch.empty(args.num_envs,agent._model.cfg.archi.z_dim,device=env.device)
            for group,ref in enumerate(refs if args.group_motion else refs[:1]):
                mask=physical_groups==group
                table=tracking_context(agent._model,ref,args.z_window,dims)
                frame=env.episode_length_buf[mask].clamp(max=len(table)-1)
                zref[mask]=table[frame]
            z=torch.where(ref_mask,zref,random_z)
            if args.evaluate:z=zref
            distribution=agent._model.actor(obs,z,agent._model.cfg.actor_std)
            action=(distribution.mean if args.evaluate else distribution.sample()).clamp(-1,1)
            if args.tactile_validation or args.reference_replay:
                q=env.ref_qpos[env.episode_length_buf.clamp(max=env.ref_len-1),0,7:]
                if args.hand_only:
                    q=q.clone()
                    for group,motion in enumerate(env.hand_motion):
                        mask=env.group_id==group
                        q[mask]=motion[env.episode_length_buf[mask].clamp(max=len(motion)-1),6:]
                action=(2*(q-bridge.term.lower)/(bridge.term.upper-bridge.term.lower)-1).clamp(-1,1)
            previous=bridge.last.clone();bridge.before_step(obs,action)
            phase=env.episode_length_buf.clone()
            if args.tactile_probe:place_probe(env,bridge.force_provider,step)
            _,_,terminated,truncated,_=wrapped.step(action)
            if height_audit is not None and step in (0,1,9,29,59,99,149,198):height_audit.capture('after_physics',step)
            if step==0:
                speed=bridge.reward_state['object_velocity'][:,:3].norm(dim=-1)
                (out/'first-formal-step.json').write_text(json.dumps({'speed_gt_1m_s_by_group':{str(int(g)):int(((physical_groups==g)&(speed>1)).sum()) for g in torch.unique(physical_groups)},'max_speed_m_s':float(speed.max())},indent=2))
            done=terminated|truncated
            assert bridge.captured is not None
            if success_log is not None:
                lengths=env.group_lengths if args.group_motion else env.ref_len
                success_log.step(phase,lengths,bridge.height-initial_z,bridge.bottom,done,step+1)
                candidate=success_log.reference_snapshot()
                if not args.collect_only and improves(candidate,best_record):
                    best_record={**candidate,'step':step+1,'total_updates':update_offset+updates}
                    # Save before the next gradient update, preserving the policy that just acted.
                    save_checkpoint('best.ckpt',step+1)
                    temporary=out/'best.json.tmp'
                    temporary.write_text(json.dumps(best_record,indent=2))
                    temporary.replace(out/'best.json')
                    writer.add_scalar('Success/best_macro_pct',100*best_record['macro_rate'],step+1)
                    print(json.dumps({'best_checkpoint':'best.ckpt',**best_record}),flush=True)
            transition={'observation':obs,'action':action,'z':z,
                'reference_context':ref_mask.clone(),'reference_frame':phase[:,None].clone(),
                'next':{'observation':bridge.captured,'terminated':done[:,None]},
                'object_id':physical_groups[:,None].clone(),
                'reward_state':bridge.reward_state,'rp':bridge.rp,'aux_rewards':{}}
            if args.group_motion:
                from torch.utils._pytree import tree_map
                for group,buffer in enumerate(group_replays):
                    buffer.add(tree_map(lambda x:x[env.group_id==group],transition))
            else:replay.add(transition)
            ids=done.nonzero().flatten();episodes+=len(ids)
            if args.evaluate:
                late=(phase>=(env.group_lengths if args.group_motion else env.ref_len)-61)&~seen
                hold_min[late]=torch.minimum(hold_min[late],bridge.height[late]-initial_z[late])
                hold_bottom[late]=torch.minimum(hold_bottom[late],bridge.bottom[late])
                for i in ids.tolist():
                    if not seen[i]:
                        evaluation.append({'env':i,'object_group':int(env.group_id[i]) if hasattr(env,'group_id') else 0,'min_hold_rise_m':float(hold_min[i]),'min_hold_bottom_clearance_m':float(hold_bottom[i]),'success':bool(hold_min[i]>=.04 and hold_bottom[i]>.01)})
                        seen[i]=True
            bridge.reset_history(ids)
            if len(ids):random_z[ids]=agent._model.sample_z(len(ids),device=env.device)
            obs=bridge.observe()
        if args.profile_phases:end_phase('sampling',step,phase_started)
        if not args.evaluate and not args.collect_only and step>=args.warmup and step%args.update_every==0:
            if args.profile_phases:phase_started=begin_phase()
            for _ in range(args.updates_per_step):
                metrics=agent.update({'expert_slicer':expert,'train':training_replay},step=update_offset+updates)
                assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
                updates+=1
            if args.profile_phases:end_phase('updates',step,phase_started)
        if step%32==0:
            assert all(torch.isfinite(v).all() for v in obs.values()),'Non-finite physical observation'
            print(json.dumps(dict(step=step,updates=updates,episodes=episodes,elapsed_s=time.monotonic()-started)),flush=True)
        if writer is not None and (step+1)%args.tb_interval==0:
            samples=(step+1)*args.num_envs
            for key,value in metrics.items():writer.add_scalar('loss_and_metrics/'+key,float(torch.as_tensor(value).float().mean()),step+1)
            writer.add_scalar('training/updates',updates,step+1)
            writer.add_scalar('training/sampling_steps',step+1,step+1)
            writer.add_scalar('training/transitions_per_second',samples/(time.monotonic()-started),step+1)
            writer.add_scalar('training/action_saturation_fraction',float((action.abs()>.99).float().mean()),step+1)
            if str(env.device).startswith('cuda'):
                writer.add_scalar('training/peak_torch_memory_gb',torch.cuda.max_memory_allocated()/1024**3,step+1)
        if not args.evaluate and (step+1==32 or (step+1)%args.save_every==0 or time.monotonic()-last_saved>=args.save_seconds):
            periodic=(step+1==32 or (step+1)%args.save_every==0)
            save_checkpoint(f'checkpoint_{step+1}.pt' if periodic else 'checkpoint_latest.pt',step+1)
            last_saved=time.monotonic()
        if stop_requested:break
        if args.evaluate and bool(seen.all()):break
        if step+1>=args.steps and (args.evaluate or time.monotonic()-started>=args.min_train_seconds):break
    actual_steps=step+1
    if not args.evaluate:save_checkpoint('checkpoint.pt',actual_steps)
    else:(out/'evaluation.json').write_text(json.dumps({'successes':sum(x['success'] for x in evaluation),'count':len(evaluation),'episodes':evaluation},indent=2))
    if args.replay_bank and not args.evaluate:
        from torch.utils._pytree import tree_map
        shard=bank/(args.object_key+'.pt');temporary=shard.with_suffix('.tmp')
        torch.save({'dimensions':dims,'object_key':args.object_key,'transitions':tree_map(lambda x:x.cpu(),replay.sample(min(replay.size,32768)))},temporary)
        temporary.replace(shard)
    report={'steps':actual_steps,'elapsed_s':time.monotonic()-started,'transitions':actual_steps*args.num_envs,'updates':updates,'episodes':episodes,'retained_objects':len(retained),
        'finite_metrics':True,'initial_corner_error_m':corner_error,'skill_success_not_evaluated':True,'metrics':{k:float(torch.as_tensor(v).float().mean()) for k,v in metrics.items()}}
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if args.tactile!='none':
        from contact_reference import LINKS
        tactile_report={'contact_order':list(LINKS),'peak_force_N':bridge.force_provider.peak.tolist(),'active_observation_counts':bridge.force_provider.active_counts.tolist(),
                        'sensor_bodies':[s.body_names for s in bridge.force_provider.sensors],
                        'filters':[s.cfg.filter_prim_paths_expr for s in bridge.force_provider.sensors],
                        'mode':args.tactile,'actor_tactile_dim':dims['tactile'],'actor_history_dim':dims['history_actor']}
        (out/'tactile_validation.json').write_text(json.dumps(tactile_report,indent=2))
except BaseException:
    import traceback
    error=traceback.format_exc();(out/'error.txt').write_text(error);print(error,flush=True)
    raise
finally:
    if writer is not None:writer.close()
    wrapped.close();app.close()
