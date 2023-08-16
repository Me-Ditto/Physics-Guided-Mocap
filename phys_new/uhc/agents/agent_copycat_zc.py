import argparse
import math
import os
import time

import fasteners
import torch

os.environ["OMP_NUM_THREADS"] = "1"
import glob
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import joblib
import wandb
from tqdm import tqdm

# from utils.adjust_utils import Adjust

# if sys.version_info >= (3, 8):

if sys.version_info >= (3, 8):
    os.add_dll_directory(os.path.abspath('data/mujoco/mjpro150/bin'))
    os.add_dll_directory(os.path.abspath('data/mujoco/mujoco-py-1.50.1.0/mujoco_py'))
# os.add_dll_directory("E://ChenZhu//UHC_shaped//data//mujoco//mjpro150//bin")
# os.add_dll_directory("E://ChenZhu//UHC_shaped//data//mujoco//mujoco-py-1.50.1.0//mujoco_py")
# else:
os.environ.setdefault('PATH', '')
os.environ['PATH'] += os.pathsep + "data/mujoco/mjpro150/bin"
os.environ['PATH'] += os.pathsep + "data/mujoco/mujoco-py-1.50.1.0/mujoco_py"

sys.path.append('./phys_new')
import multiprocessing

from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.envs.humanoid_im import HumanoidEnv
from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.rl.agents import AgentPPO
from uhc.khrylib.rl.core import LoggerRL, estimate_advantages
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.utils import ZFilter, create_logger, get_eta_str, to_device
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.utils.torch import *
from uhc.losses.reward_function import reward_func
from uhc.models.policy_mcp import PolicyMCP
from uhc.smpllib.smpl_eval import compute_metrics
from uhc.smpllib.smpl_mujoco import SMPL_M_Viewer, smpl_to_qpose
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES
from uhc.utils.config_utils.copycat_config import Config
from uhc.utils.flags import flags
from uhc.utils.tools import CustomUnpickler
from uhc.utils.transform_utils import (convert_aa_to_orth6d,
                                       convert_orth_6d_to_aa, rot6d_to_rotmat,
                                       rotation_matrix_to_angle_axis,
                                       vertizalize_smpl_root)

from utils.rotation_conversions import *


def seed_worker(env, pid):
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        np.random.seed(np.random.randint(5000)* pid)
        if hasattr(env, 'np_random'):
            env.np_random.random(np.random.randint(5000 )* pid)

def sample_cmd(env,pid,data):
    print(f'\n sample processid ={pid} ')
    min_batch_size,data_loader,fit_single_key,freq_dict, cfg, precision_mode, running_state, mean_action_ori, noise_rate,policy_net,custom_reward, end_reward,render= data
    seed_worker(env,pid)
    if hasattr(env, "np_random"):
        env.np_random.random(pid)
    memory = Memory()
    logger = LoggerRL()
    freq_dict_temp = defaultdict(list)
    while logger.num_steps < min_batch_size:
        # print(f'process:{pid},logger.num_steps:{logger.num_steps}')
        if fit_single_key != "":
            env.load_expert(data_loader.get_sample_from_key(
                fit_single_key,
                full_sample=False,
                freq_dict=freq_dict,
                precision_mode=precision_mode,
            ))
        else:
            env.load_expert(data_loader.sample_seq(
                freq_dict=freq_dict,
                full_sample=False,
                sampling_temp=cfg.sampling_temp,
                sampling_freq=cfg.sampling_freq,
                precision_mode=precision_mode,
            ))
        # self.env.load_expert(self.data_loader.sample_seq(freq_dict = self.freq_dict, full_sample = True))

        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        logger.start_episode(env)
        # self.pre_episode()

        for t in range(10000):
            # print(t)
            state_var = tensor(state).unsqueeze(0)
            # trans_out = self.trans_policy(state_var)
            trans_out = state_var
            mean_action = mean_action_ori or env.np_random.binomial(1, 1 - noise_rate)
            action = policy_net.select_action(trans_out, mean_action)[0].numpy()
            action = (int(action) if policy_net.type == "discrete" else action.astype(np.float64))
            next_state, env_reward, done, info = env.step(action)
            if running_state is not None:
                next_state = running_state(next_state)
            # use custom or env reward
            if custom_reward is not None:
                c_reward, c_info = custom_reward(env, state, action, info)
                reward = c_reward
            else:
                c_reward, c_info = 0.0, np.array([0.0])
                reward = env_reward

            # add end reward
            if end_reward and info.get("end", False):
                reward += env.end_reward
            # logging
            logger.step(env, env_reward, c_reward, c_info, info)

            mask = 0 if done else 1
            exp = 1 - mean_action
            memory.push(state, action, mask, next_state, reward, exp)

            if pid == 0 and render:
                for i in range(10):
                    # for i in range(500):
                    env.render()

            if done:

                freq_dict_temp[data_loader.curr_key].append([info["percent"], data_loader.fr_start])
                break
            state = next_state

        logger.end_episode(env)
    logger.end_sampling()
    
    res = [pid,memory,logger,freq_dict_temp]
    return res

def eval_seqs_cmd(env,pid,data,reload_robot=True):
    print(f'\n eval processid ={pid} ')
    job,p_sample_modules,running_state,policy_net,cc_cfg,cfg ,render,custom_reward    = data
    ress = {}
    for seq in job:
        print(f"eval seq:{seq['seq_name']} in process{pid}")                            
        env.set_mode("test")
        fail_safe = False
        with to_cpu(p_sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                # expertinput = loader.get_sample_from_key(take_key=take_key, full_sample=True, fr_start=-1)
                
                env.load_expert(seq,reload_robot=reload_robot) 

                state = env.reset()
                if running_state is not None:
                    state = running_state(state)

                for t in range(10000):
                    res["gt"].append(env.get_expert_attr("qpos", env.get_expert_index(t)).copy())

                    res["pred"].append(env.data.qpos.copy())

                    res["gt_jpos"].append(env.get_expert_attr("wbpos", env.get_expert_index(t)).copy())
                    res["pred_jpos"].append(env.get_wbody_pos().copy())
                    state_var = tensor(state).unsqueeze(0)
                    # trans_out = self.trans_policy(state_var)
                    trans_out = state_var
                    action = (policy_net.select_action(trans_out, mean_action=True)[0].cpu().numpy())
                    next_state, env_reward, done, info = env.step(action)
                    if (cc_cfg.residual_force and cc_cfg.residual_force_mode == "explicit"):
                        res["vf_world"].append(env.get_world_vf())

                    if render:
                        for i in range(10):
                            env.render()

                    c_reward, c_info = custom_reward(env, state, action, info)
                    # np.set_printoptions(precision=4, suppress=1)
                    # print(c_info)
                    res["reward"].append(c_reward)
                    # self.env.render()
                    if running_state is not None:
                        next_state = running_state(next_state, update=False)

                    if done:#摔倒或者结束  info["percent"] == 1表示结束
                        if cfg.fail_safe and info["percent"] != 1:
                            env.fail_safe() #重置？？？
                            fail_safe = True
                        else:
                            res = {k: np.vstack(v) for k, v in res.items()}
                            res["percent"] = info["percent"]
                            res["fail_safe"] = fail_safe
                            if cfg.get("full_eval", False):
                                env.convert_2_smpl_params(res)
                            res.update(compute_metrics(res, env.converter))
                            ress[seq['seq_name']] = res
                            break
                    state = next_state
                # print('debug:range10000done')
    return ress

def eval_seqs_cmd_old(env,pid,data):
    print(f'\n eval processid ={pid} ')
    take_keys,loader,p_sample_modules,running_state,policy_net,cc_cfg,cfg ,render,custom_reward    = data
    ress = {}
    for take_key in take_keys:                            
        env.set_mode("test")
        fail_safe = False
        with to_cpu(p_sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                expertinput = loader.get_sample_from_key(take_key=take_key, full_sample=True, fr_start=-1)
                
                env.load_expert(loader.get_sample_from_key(take_key=take_key, full_sample=True, fr_start=-1),reload_robot=True) 

                state = env.reset()
                if running_state is not None:
                    state = running_state(state)

                for t in range(10000):
                    res["gt"].append(env.get_expert_attr("qpos", env.get_expert_index(t)).copy())

                    res["pred"].append(env.data.qpos.copy())

                    res["gt_jpos"].append(env.get_expert_attr("wbpos", env.get_expert_index(t)).copy())
                    res["pred_jpos"].append(env.get_wbody_pos().copy())
                    state_var = tensor(state).unsqueeze(0)
                    # trans_out = self.trans_policy(state_var)
                    trans_out = state_var
                    action = (policy_net.select_action(trans_out, mean_action=True)[0].cpu().numpy())
                    next_state, env_reward, done, info = env.step(action)
                    if (cc_cfg.residual_force and cc_cfg.residual_force_mode == "explicit"):
                        res["vf_world"].append(env.get_world_vf())

                    if render:
                        for i in range(10):
                            env.render()

                    c_reward, c_info = custom_reward(env, state, action, info)
                    # np.set_printoptions(precision=4, suppress=1)
                    # print(c_info)
                    res["reward"].append(c_reward)
                    # self.env.render()
                    if running_state is not None:
                        next_state = running_state(next_state, update=False)

                    if done:#摔倒或者结束  info["percent"] == 1表示结束
                        if cfg.fail_safe and info["percent"] != 1:
                            env.fail_safe() #重置？？？
                            fail_safe = True
                        else:
                            res = {k: np.vstack(v) for k, v in res.items()}
                            res["percent"] = info["percent"]
                            res["fail_safe"] = fail_safe
                            if cfg.get("full_eval", False):
                                env.convert_2_smpl_params(res)
                            res.update(compute_metrics(res, env.converter))
                            # return res
                            ress[take_key] = res
                            break
                    state = next_state
                # print('debug:range10000done')
    return ress


def sample_worker(pid, pipe, env_cfg):
    """ run physics simulator in the subprocess """
    env = HumanoidEnv(
        env_cfg[0],
        init_expert=env_cfg[1],
        data_specs=env_cfg[2],
        mode=env_cfg[3],
        no_root=env_cfg[4],
    )
    pipe.send(pid)
    while True:
        try:
            cmd, data = pipe.recv()
        except KeyboardInterrupt:
            print(f"subprocess {pid} worker: got KeyboardInterrupt")
            break
        
        if cmd == "sample":
            res = sample_cmd(env,pid,data)
            pipe.send(res)
            # print(f'sample : subprocessing{pid} over')
        elif cmd == "eval_seqs":
            ress = eval_seqs_cmd(env,pid,data)
            pipe.send(ress)
            # print(f'eval_seqs : subprocessing{pid} over')
        elif cmd == "close":
            pipe.send([f"subprocess {pid} close"])
            break
        else:
            raise RuntimeError(f"Got unrecognized cmd {cmd}")
    pipe.close()



class AgentCopycat_zc(AgentPPO):

    def __init__(self,args=None, dtype=None, device=None, training=True, checkpoint_epoch=0, num_threads=1):
        if args is None:
            sys.argv = [
            "",
            "--cfg=uhc_implicit",
            "--epoch=19000",
            "--data=data/sample_data/amass_copycat_take5_test_small.pkl",
            "--test",
            "--render_video"
            ]
            parser = argparse.ArgumentParser()
            parser.add_argument("--cfg", default=None)
            parser.add_argument("--test", action="store_true", default=False)
            parser.add_argument("--num_threads", type=int, default=1)
            parser.add_argument("--gpu_index", type=int, default=0)
            parser.add_argument("--epoch", type=int, default=0)
            parser.add_argument("--show_noise", action="store_true", default=False)
            parser.add_argument("--resume", type=str, default=None)
            parser.add_argument("--no_log", action="store_true", default=False)
            parser.add_argument("--debug", action="store_true", default=False)
            parser.add_argument("--data", type=str, default="data/sample_data/amass_copycat_take5_test_small.pkl")
            parser.add_argument("--mode", type=str, default="all") # vis 
            parser.add_argument("--render_video", action="store_true", default=False)
            parser.add_argument("--render_rfc", action="store_true", default=False)
            parser.add_argument("--render", action="store_true", default=False)
            parser.add_argument("--hide_expert", action="store_true", default=False)
            parser.add_argument("--no_fail_safe", action="store_true", default=False)
            parser.add_argument("--focus", action="store_true", default=False)
            parser.add_argument("--output", type=str, default="test")
            parser.add_argument("--shift_expert", action="store_true", default=False)
            parser.add_argument("--smplx", action="store_true", default=False)
            parser.add_argument("--hide_im", action="store_true", default=False)
            parser.add_argument("--adjust", action="store_true", default=False)
            args = parser.parse_args()

            # args.cfg = 'uhc_implicit' 
            # args.epoch = 19000
            # args.data = 'data/sample_data/amass_copycat_take5_test_small.pkl'
            # args.test = True
            args.num_threads = num_threads
            
        if dtype is None :
            dtype = torch.float64
        if device is None:
            torch.set_default_dtype(dtype)
            device = (
                torch.device("cuda", index=args.gpu_index)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            # device = torch.device("cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_index)
            print(f"Using: {device}")

        cfg = self.config_init(args)
        self.cfg = cfg
        self.cc_cfg = cfg
        self.device = device
        self.dtype = dtype
        self.training = training
        self.max_freq = 50

        self.setup_vars()
        self.setup_data_loader() #finish amass class data
        self.setup_env()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()# agent copycat
        if checkpoint_epoch==0:
            checkpoint_epoch = args.epoch
        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)
            self.epoch = checkpoint_epoch

        super().__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            running_state=self.running_state,
            custom_reward=self.expert_reward,
            mean_action=cfg.render and not cfg.show_noise,
            render=cfg.render,
            num_threads=cfg.num_threads,
            data_loader=self.data_loader,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.num_optim_epoch,
            gamma=cfg.gamma,
            tau=cfg.tau,
            clip_epsilon=cfg.clip_epsilon,
            policy_grad_clip=[(self.policy_net.parameters(), 40)],
            end_reward=cfg.end_reward,
            use_mini_batch=False,
            mini_batch_size=0,
        )
        self.tasks = []
        #多进程及通道备用
        """ multiprocessing """
        if self.num_threads>1:
            num_processes = self.num_threads
            self._num_processes = num_processes
            self._processes = []
            self._parent_pipes = []
            self._child_pipes = []
            cfg, device, dtype = self.cfg, self.device, self.dtype
            # random_expert_1 = self.data_loader.sample_seq()
            random_expert_1 = joblib.load(self.cfg.data_specs["init_path"])
            env_cfg =[cfg,random_expert_1,cfg.data_specs,"train",cfg.no_root]        
            for _ in range(num_processes-1):
                parent, child = multiprocessing.Pipe()
                self._parent_pipes.append(parent)
                self._child_pipes.append(child)
            for i in range(num_processes-1):
                pid = i+1 
                process = multiprocessing.Process(target=sample_worker, args=(pid, self._child_pipes[i], env_cfg))
                process.start()
                self._processes.append(process)
            for i, parentPipe in enumerate(self._parent_pipes):
                print(f'process{parentPipe.recv()} init done!')    
            print('all process pipe prepared')


    def config_init(self, args):
        cfg = Config(cfg_id=args.cfg, create_dirs=False)
        cfg.update(args)

        if cfg.test: 
            cfg.no_log = True
            if args.no_fail_safe:
                cfg.fail_safe = False

            cfg.output = osp.join("self_output", f"{cfg.id}")
            os.makedirs(cfg.output, exist_ok=True)

            cfg.data_specs["file_path"] = args.data

            if "test_file_path" in cfg.data_specs:
                del cfg.data_specs["test_file_path"]

            if cfg.mode == "vis":
                cfg.num_threads = 1

        return cfg

    def setup_vars(self):
        self.epoch = 0
        self.running_state = None
        self.fit_single_key = ""
        self.precision_mode = self.cc_cfg.get("precision_mode", False)

        pass

    def print_config(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.logger.info("==========================Agent Copycat===========================")
        self.logger.info(f"Feature Version: {cfg.obs_v}")
        self.logger.info(f"Meta Pd: {cfg.meta_pd}")
        self.logger.info(f"Meta Pd Joint: {cfg.meta_pd_joint}")
        self.logger.info(f"Actor_type: {cfg.actor_type}")
        self.logger.info(f"Precision mode: {self.precision_mode}")
        self.logger.info(f"State_dim: {self.state_dim}")
        self.logger.info("============================================================")

    def setup_data_loader(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.data_loader = data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="train")
        self.test_data_loaders = []
        self.test_data_loaders.append(data_loader)
        if len(cfg.data_specs.get("test_file_path", [])) > 0:
            self.test_data_loaders.append(DatasetAMASSSingle(cfg.data_specs, data_mode="test"))

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        # aaa =  joblib.load(self.cfg.data_specs["init_path"])
        random_expert = self.data_loader.sample_seq()
        self.env = HumanoidEnv(
            cfg,
            # init_expert=random_expert,
            init_expert= joblib.load(self.cfg.data_specs["init_path"]),
            # init_expert= joblib.load(open(self.cfg.data_specs["init_path"],"rb")),
            data_specs=cfg.data_specs,
            mode="train",
            no_root=cfg.no_root,
        )

    def setup_policy(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        actuators = env.model.actuator_names
        self.state_dim = state_dim = env.observation_space.shape[0]
        self.action_dim = action_dim = env.action_space.shape[0]
        """define actor and critic"""
        if cfg.actor_type == "gauss":
            self.policy_net = PolicyGaussian(cfg, action_dim=action_dim, state_dim=state_dim)
        elif cfg.actor_type == "mcp":
            self.policy_net = PolicyMCP(cfg, action_dim=action_dim, state_dim=state_dim)
        self.running_state = ZFilter((state_dim,), clip=5)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
        to_device(device, self.value_net)

    def setup_optimizer(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        if cfg.policy_optimizer == "Adam":
            self.optimizer_policy = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=cfg.policy_lr,
                weight_decay=cfg.policy_weightdecay,
            )
        else:
            self.optimizer_policy = torch.optim.SGD(
                self.policy_net.parameters(),
                lr=cfg.policy_lr,
                momentum=cfg.policy_momentum,
                weight_decay=cfg.policy_weightdecay,
            )
        if cfg.value_optimizer == "Adam":
            self.optimizer_value = torch.optim.Adam(
                self.value_net.parameters(),
                lr=cfg.value_lr,
                weight_decay=cfg.value_weightdecay,
            )
        else:
            self.optimizer_value = torch.optim.SGD(
                self.value_net.parameters(),
                lr=cfg.value_lr,
                momentum=cfg.value_momentum,
                weight_decay=cfg.value_weightdecay,
            )

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = expert_reward = reward_func[cfg.reward_id]

    def save_checkpoint(self, epoch):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, epoch + 1)
            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path, "wb"))
            joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def save_singles(self, epoch, key):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = f"{cfg.model_dir}_singles/{key}.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path, "wb"))

    def save_curr(self):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path_best = f"{cfg.model_dir}/iter_best.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path_best, "wb"))

    def load_curr(self):
        cfg = self.cfg
        cp_path_best = f"{cfg.model_dir}/iter_best.p"
        self.logger.info("loading model from checkpoint: %s" % cp_path_best)
        model_cp = CustomUnpickler(open(cp_path_best, "rb")).load()
        self.policy_net.load_state_dict(model_cp["policy_dict"])
        self.value_net.load_state_dict(model_cp["value_dict"])
        self.running_state = model_cp["running_state"]

    def load_singles(self, epoch, key):
        cfg = self.cfg
        # self.tb_logger.flush()
        if epoch > 0:
            cp_path = f"{cfg.model_dir}/iter_{(epoch+ 1):04d}_{key}.p"
            self.logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])
            self.value_net.load_state_dict(model_cp["value_dict"])
            self.running_state = model_cp["running_state"]

    def load_checkpoint(self, iter):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        cfg = self.cfg
        if iter > 0:
            # cp_path = "%s/iter_%04d.p" % (cfg.model_dir, iter)
            cp_path = "%s/iter_%04d.p" % ('data', iter) 
            self.logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])
            self.value_net.load_state_dict(model_cp["value_dict"])
            self.running_state = model_cp["running_state"]

        to_device(device, self.policy_net, self.value_net)

    def setup_logging(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        freq_path = osp.join(cfg.result_dir, "freq_dict.pt")
        try:
            self.freq_dict = ({k: [] for k in self.data_loader.data_keys} if not osp.exists(freq_path) else joblib.load(freq_path))
            for k in self.data_loader.data_keys:
                if not k in self.freq_dict:
                    raise Exception("freq_dict is not initialized")

            for k in self.freq_dict:
                if not k in self.data_loader.data_keys:
                    raise Exception("freq_dict is not initialized")
        except:
            print("error parsing freq_dict, using empty one")
            self.freq_dict = {k: [] for k in self.data_loader.data_keys}
        self.logger = create_logger(os.path.join(cfg.log_dir, "log.txt"))

    def per_epoch_update(self, epoch):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        cfg.update_adaptive_params(epoch)
        self.set_noise_rate(cfg.adp_noise_rate)
        set_optimizer_lr(self.optimizer_policy, cfg.adp_policy_lr)
        if cfg.rfc_decay:
            if self.epoch < cfg.get("rfc_decay_max", 10000):
                self.env.rfc_rate = lambda_rule(self.epoch, cfg.get("rfc_decay_max", 10000), cfg.num_epoch_fix)
            else:
                self.env.rfc_rate = 0.0

        # epoch
        # adative_iter = cfg.data_specs.get("adaptive_iter", -1)
        # if epoch != 0 and adative_iter != -1 and epoch % adative_iter == 0 :
        # agent.data_loader.hard_negative_mining(agent.value_net, agent.env, device, dtype, running_state = running_state, sampling_temp = cfg.sampling_temp)

        if cfg.fix_std:
            self.policy_net.action_log_std.fill_(cfg.adp_log_std)
        return

    def log_train(self, info):
        """logging"""
        cfg, device, dtype = self.cfg, self.device, self.dtype
        log = info["log"]

        c_info = log.avg_c_info
        log_str = f"Ep: {self.epoch}\t {cfg.id} \tT_s {info['T_sample']:.2f}\t \
                    T_u { info['T_update']:.2f}\tETA {get_eta_str(self.epoch, cfg.num_epoch, info['T_total'])} \
                \texpert_R_avg {log.avg_c_reward:.4f} {np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=',')}\
                 \texpert_R_range ({log.min_c_reward:.4f}, {log.max_c_reward:.4f})\teps_len {log.avg_episode_len:.2f}"

        self.logger.info(log_str)

        if not cfg.no_log:
            wandb.log(
                data={
                    "rewards": log.avg_c_info,
                    "eps_len": log.avg_episode_len,
                    "avg_rwd": log.avg_c_reward,
                    "rfc_rate": self.env.rfc_rate,
                },
                step=self.epoch,
            )

            if "log_eval" in info:
                [wandb.log(data=test, step=self.epoch) for test in info["log_eval"]]

    def optimize_policy(self, epoch, save_model=True):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.epoch = epoch
        t0 = time.time()
        self.per_epoch_update(epoch)
        batch, log = self.sample(cfg.min_batch_size)

        if cfg.end_reward:
            self.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {
            "log": log,
            "T_sample": t1 - t0,
            "T_update": t2 - t1,
            "T_total": t2 - t0,
        }

        if save_model and (self.epoch + 1) % cfg.save_n_epochs == 0:
            self.save_checkpoint(epoch)
            log_eval = self.eval_policy(epoch)
            info["log_eval"] = log_eval

        self.log_train(info)
        # joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def eval_policy_OLD(self, epoch=0, dump=False):
        cfg = self.cfg
        data_loaders = self.test_data_loaders

        res_dicts = []
        for data_loader in data_loaders:
            # num_jobs = 20
            num_jobs = self.num_threads
            jobs = data_loader.data_keys
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
            data_res_coverage = {}
            with to_cpu(*self.sample_modules):
                with torch.no_grad():
                    # send to subprocess
                    for i,job in enumerate(range(len(jobs)-1)):
                        pid = i+1
                        # data = [take_keys,loader,p_sample_modules,running_state,policy_net,cc_cfg,cfg ,render,custom_reward]
                        data = [jobs[pid], data_loader,*self.sample_modules,self.running_state,self.policy_net,self.cc_cfg,self.cfg ,self.render,self.custom_reward]
                        self._parent_pipes[i].send(["eval_seqs", data])
                    data = [jobs[0], data_loader,*self.sample_modules,self.running_state,self.policy_net,self.cc_cfg,self.cfg ,self.render,self.custom_reward]
                    res = eval_seqs_cmd(self.env,0,data)
                    data_res_coverage.update(res)
                    # results = []
                    # receive data
                    for i, job in enumerate(range(len(jobs)-1)):
                        res = self._parent_pipes[i].recv()
                        data_res_coverage.update(res)
                        # results.append(parentPipe.recv()) 
                        
                for k, res in data_res_coverage.items():
                    [self.freq_dict[k].append([res["succ"][0], 0]) for _ in range(1 if res["succ"][0] else 3) if k in self.freq_dict]  # first item is scuccess or not, second indicates the frame number

                metric_names = [
                    "mpjpe",
                    "mpjpe_g",
                    "accel_dist",
                    "vel_dist",
                    "succ",
                    "reward",
                    "root_dist",
                    "pentration",
                    "skate",
                ]
                data_res_metrics = defaultdict(list)
                [[data_res_metrics[k].append(v if np.ndim(v) == 0 else np.mean(v)) for k, v in res.items() if k in metric_names] for k, res in data_res_coverage.items()]
                data_res_metrics = {k: np.mean(v) for k, v in data_res_metrics.items()}
                coverage = int(data_res_metrics["succ"] * data_loader.get_len())
                print_str = " \t".join([f"{k}: {v:.3f}" for k, v in data_res_metrics.items()])

                self.logger.info(f"Coverage {data_loader.name} of {coverage} out of {data_loader.get_len()} | {print_str}")
                data_res_metrics.update({
                    "mean_coverage": coverage / data_loader.get_len(),
                    "num_coverage": coverage,
                    "all_coverage": data_loader.get_len(),
                })
                del data_res_metrics["succ"]
                res_dicts.append({f"coverage_{data_loader.name}": data_res_metrics})

                if dump:
                    res_dir = osp.join(cfg.output_dir, f"{epoch}_{data_loader.name}_coverage_full.pkl")
                    print(res_dir)
                    joblib.dump(data_res_coverage, res_dir)

        return res_dicts

    def eval_policy(self, epoch=0, dump=False):
        cfg = self.cfg

        res_dicts = []
        for task in self.tasks:
            # num_jobs = 3
            num_jobs = self.num_threads
            jobs = list(range(len(task)))
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
            
            data_res_coverage = {}
            with to_cpu(*self.sample_modules):
                with torch.no_grad():
                    # send to subprocess
                    for i in range(len(jobs)-1):
                        pid = i+1
                        job = [task[i] for i in jobs[pid]]
                        # data = [take_keys,loader,p_sample_modules,running_state,policy_net,cc_cfg,cfg ,render,custom_reward]
                        data = [job,*self.sample_modules,self.running_state,self.policy_net,self.cc_cfg,self.cfg ,self.render,self.custom_reward]
                        self._parent_pipes[i].send(["eval_seqs", data])
                    job = [task[i] for i in jobs[0]]
                    data = [job,*self.sample_modules,self.running_state,self.policy_net,self.cc_cfg,self.cfg ,self.render,self.custom_reward]
                    res = eval_seqs_cmd(self.env,0,data)
                    data_res_coverage.update(res)
                    # results = []
                    # receive data
                    for i in range(len(jobs)-1):
                        res = self._parent_pipes[i].recv()
                        data_res_coverage.update(res)
                        # results.append(parentPipe.recv()) 
                        
                for k, res in data_res_coverage.items():
                    [self.freq_dict[k].append([res["succ"][0], 0]) for _ in range(1 if res["succ"][0] else 3) if k in self.freq_dict]  # first item is scuccess or not, second indicates the frame number

                metric_names = [
                    "mpjpe",
                    "mpjpe_g",
                    "accel_dist",
                    "vel_dist",
                    "succ",
                    "reward",
                    "root_dist",
                    "pentration",
                    "skate",
                ]
                data_res_metrics = defaultdict(list)
                [[data_res_metrics[k].append(v if np.ndim(v) == 0 else np.mean(v)) for k, v in res.items() if k in metric_names] for k, res in data_res_coverage.items()]
                data_res_metrics = {k: np.mean(v) for k, v in data_res_metrics.items()}
                coverage = int(data_res_metrics["succ"] * len(self.tasks[0]))
                print_str = " \t".join([f"{k}: {v:.3f}" for k, v in data_res_metrics.items()])

                self.logger.info(f"Coverage task id={0} of {coverage} out of {len(self.tasks[0])} | {print_str}")
                data_res_metrics.update({
                    "mean_coverage": coverage / len(self.tasks[0]),
                    "num_coverage": coverage,
                    "all_coverage": len(self.tasks[0]),
                })
                del data_res_metrics["succ"]
                res_dicts.append({f"coverage_task id={0}": data_res_metrics})

                if dump:
                    res_dir = osp.join(cfg.output_dir, f"{epoch}_task id={0}_coverage_full.pkl")
                    print(res_dir)
                    joblib.dump(data_res_coverage, res_dir)

        return res_dicts, data_res_coverage


    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)


    def eval_cur_seq(self):
        return self.eval_seq(self.fit_ind, self.data_loader)

    def eval_seq(self, take_key, loader):
        self.env.set_mode("test")
        fail_safe = False
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)

                self.env.load_expert(loader.get_sample_from_key(take_key=take_key, full_sample=True, fr_start=-1),reload_robot=True) 

                state = self.env.reset()
                if self.running_state is not None:
                    state = self.running_state(state)

                for t in range(10000):
                    res["gt"].append(self.env.get_expert_attr("qpos", self.env.get_expert_index(t)).copy())

                    res["pred"].append(self.env.data.qpos.copy())

                    res["gt_jpos"].append(self.env.get_expert_attr("wbpos", self.env.get_expert_index(t)).copy())
                    res["pred_jpos"].append(self.env.get_wbody_pos().copy())
                    state_var = tensor(state).unsqueeze(0)
                    trans_out = self.trans_policy(state_var)

                    action = (self.policy_net.select_action(trans_out, mean_action=True)[0].cpu().numpy())
                    next_state, env_reward, done, info = self.env.step(action)
                    if (self.cc_cfg.residual_force and self.cc_cfg.residual_force_mode == "explicit"):
                        res["vf_world"].append(self.env.get_world_vf())

                    if self.render:
                        for i in range(10):
                            self.env.render()

                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    # np.set_printoptions(precision=4, suppress=1)
                    # print(c_info)
                    res["reward"].append(c_reward)
                    # self.env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state, update=False)

                    if done:
                        if self.cfg.fail_safe and info["percent"] != 1:
                            self.env.fail_safe()
                            fail_safe = True
                        else:
                            res = {k: np.vstack(v) for k, v in res.items()}
                            res["percent"] = info["percent"]
                            res["fail_safe"] = fail_safe
                            if self.cfg.get("full_eval", False):
                                self.env.convert_2_smpl_params(res)
                            res.update(compute_metrics(res, self.env.converter))
                            return res
                    state = next_state
                print('debug:range10000done')

    def sample(self, min_batch_size):
        # self.env.set_mode("train")
        t_start = time.time()
        self.pre_sample()
        to_test(*self.sample_modules)
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads

                # send to subprocess
                for i, parentPipe in enumerate(self._parent_pipes):
                    data = [thread_batch_size,self.data_loader,self.fit_single_key,self.freq_dict, self.cfg, self.precision_mode, self.running_state, self.mean_action, self.noise_rate,self.policy_net,self.custom_reward, self.end_reward,self.render]
                    parentPipe.send(["sample", data])
                
                res0 = sample_cmd(self.env,0,data)
                results = []
                results.append(res0)
                # receive data
                for i, parentPipe in enumerate(self._parent_pipes):
                     results.append(parentPipe.recv()) 
                
                for i in range(self.num_threads):
                    pid, worker_memory, worker_logger, freq_dict = results[i]
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                    self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}
                # print('sample ok!')

                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers)

        self.freq_dict = {k: v if len(v) < self.max_freq else v[-self.max_freq:] for k, v in self.freq_dict.items()}
        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def trans_smpl_sim(self, smpl_pose, smpl_trans):
        pose_aa = smpl_pose
        curr_trans = smpl_trans
        # pose_seq_6d = convert_aa_to_orth6d(torch.tensor(pose_aa.clone().detach())).reshape(-1, 144)
        pose_seq_6d = convert_aa_to_orth6d(pose_aa.clone().detach()).reshape(-1, 144)
        
        qpos = smpl_to_qpose(pose = pose_aa, mj_model = self.env.sim_model, trans = curr_trans)
        return {
            "pose_aa":pose_aa.squeeze().numpy(), 
            "pose_6d":pose_seq_6d.squeeze().numpy(), 
            "qpos":qpos[0], 
            "trans":curr_trans.squeeze().numpy()
        }
    
    def formate_experts(self, pklpath=R'E:\ChenZhu\data\h36m_annot\test_clip0_10.pkl'):
        #for h36m
        print('loading h36m pkl')
        data = joblib.load(pklpath)
        # print('debug')
        out_expert = []
        for seq in tqdm(data):
            charactername = seq[0]['img_path'].split('/')[1]
            motionname = seq[0]['img_path'].split('/')[2]
            camname = seq[0]['img_path'].split('/')[3]
            seqname = charactername + '__' + motionname + '__' + camname
            
            for i, item in enumerate(seq):
                
                smpl_pose = torch.from_numpy(np.array(item['0']['pose'], dtype=np.float32)).reshape(-1, 72)
                smpl_trans = torch.from_numpy(np.array(item['0']['trans'], dtype=np.float32)).reshape(-1, 3)
                smpl_betas = np.array(item['0']['betas'], dtype=np.float32)

                if self.cfg.adjust:
                    
                    if (i == 0):
                        pass
                        # adjust = Adjust(pose, trans, 30, smpl, 0)
                    else:
                        pass
                        # pose, trans = adjust(pose, trans)

                sim_data = self.trans_smpl_sim(smpl_pose.clone().detach(), smpl_trans.clone().detach())
                if i==0:
                    ini_expert = {
                    'pose_aa':sim_data['pose_aa'][None, :], 
                    'pose_6d':sim_data['pose_6d'][None, :], 
                    'trans':sim_data['trans'][None, :], 
                    'beta':smpl_betas, 
                    # 'gender':np.array([0]) ,                      
                    'obj_pose':sim_data['qpos'][None, :], 
                    'seq_name':seqname, 
                    'has_obj':False, 
                    'num_obj':0
                    }
                else:
                    ini_expert['pose_aa'] = np.vstack([ini_expert['pose_aa'], sim_data['pose_aa'][None, :]])
                    ini_expert['pose_6d'] = np.vstack([ini_expert['pose_6d'], sim_data['pose_6d'][None, :]])
                    ini_expert['trans'] = np.vstack([ini_expert['trans'], sim_data['trans'][None, :]])
                    ini_expert['obj_pose'] = np.vstack([ini_expert['obj_pose'], sim_data['qpos'][None, :]])
                    ini_expert['beta'] = np.vstack([ini_expert['beta'], smpl_betas[None, :]])

            ini_expert['gender'] =  np.array([0]*(i+1))
            out_expert.append(ini_expert)
        self.tasks.append(out_expert)
        savepath = self.cfg.result_dir + '\\' +'save_experts.pkl'
        joblib.dump(out_expert, savepath)
        print(f'experts stored in {savepath}')



    def cut_pkl(self, pklpath=R'E:\ChenZhu\data\h36m_annot\test.pkl', t=0):
        data = joblib.load(pklpath)
        temp = []
        for i, item in enumerate(data):
            temp.append(item)
            if i==t:
                break
        joblib.dump(temp, pklpath.rstrip('.pkl')+f'_clip0_{t}.pkl')

    def load_data(self, pklpath=R'E:\ChenZhu\data\h36m_annot\test_clip0_10.pkl'):
        print(f'loading data:{pklpath}')
        data = joblib.load(pklpath)
        # print('debug')
        alldata = []
        for seq in tqdm(data):
            charactername = seq[0]['img_path'].split('/')[1]
            motionname = seq[0]['img_path'].split('/')[2]
            camname = seq[0]['img_path'].split('/')[3]
            seqname = charactername + '__' + motionname + '__' + camname
            poselist=[]
            translist=[]
            beta = np.array(seq[0]['0']['betas'], dtype=np.float32)
            for i, item in enumerate(seq):
                poselist.append(np.array(item['0']['pose'], dtype=np.float32).reshape(-1, 72))
                translist.append(np.array(item['0']['trans'], dtype=np.float32).reshape(-1, 3))
            alldata.append(dict(seqname=seqname, pose=poselist, trans=translist, beta=beta))
            # break #debug
        return alldata
    
    def load_single_data(self, pklpath=R'E:\ChenZhu\data\h36m_annot\test_clip0_10.pkl'):
        print(f'loading data:{pklpath}')
        data = joblib.load(pklpath)
        # charactername = data['img_path'].split('/')[1]
        # motionname = data['img_path'].split('/')[2]
        # camname = data['img_path'].split('/')[3]
        # seqname = charactername + '__' + motionname + '__' + camname
        poselist=[]
        translist=[]
        beta = np.array(data['0']['betas'], dtype=np.float32)
        for i, item in enumerate(data):
            poselist.append(np.array(item['0']['pose'], dtype=np.float32).reshape(-1, 72))
            translist.append(np.array(item['0']['trans'], dtype=np.float32).reshape(-1, 3))
        result = dict(pose=poselist, trans=translist, beta=beta)
        # break #debug
        return result


    def smpl2expert(self, pose, trans, beta, seqname): #input np_data       
        '''
            pose:[ np.pose(-1, 72) , ... ]
            trans:[ np.trans(-1, 3), ... ]
            beta: np.beta(10, )
            seqname: s11__xxx__cam_x
        '''
        for i, item in enumerate(pose):
            smpl_pose = torch.from_numpy(pose[i]).reshape(-1, 72)
            smpl_trans = torch.from_numpy(trans[i]).reshape(-1, 3)
            smpl_betas = beta

            if self.cfg.adjust:
                
                if (i == 0):
                    pass
                    # adjust = Adjust(pose, trans, 30, smpl, 0)
                else:
                    pass
                    # pose, trans = adjust(pose, trans)

            sim_data = self.trans_smpl_sim(smpl_pose.clone().detach(), smpl_trans.clone().detach())
            if i==0:
                ini_expert = {
                'pose_aa':sim_data['pose_aa'][None, :], 
                'pose_6d':sim_data['pose_6d'][None, :], 
                'trans':sim_data['trans'][None, :], 
                'beta':smpl_betas, 
                # 'gender':np.array([0]) ,                      
                'obj_pose':sim_data['qpos'][None, :], 
                'seq_name':seqname, 
                'has_obj':False, 
                'num_obj':0
                }
            else:
                ini_expert['pose_aa'] = np.vstack([ini_expert['pose_aa'], sim_data['pose_aa'][None, :]])
                ini_expert['pose_6d'] = np.vstack([ini_expert['pose_6d'], sim_data['pose_6d'][None, :]])
                ini_expert['trans'] = np.vstack([ini_expert['trans'], sim_data['trans'][None, :]])
                ini_expert['obj_pose'] = np.vstack([ini_expert['obj_pose'], sim_data['qpos'][None, :]])
                ini_expert['beta'] = np.vstack([ini_expert['beta'], smpl_betas[None, :]])
        ini_expert['gender'] =  np.array([0]*(i+1))
        return ini_expert

        # print('debug')

    def simulate(self, pose, trans, beta=None, seqname=None):
        '''
            pose:[ np.pose(-1, 72) , ... ]
            trans:[ np.trans(-1, 3), ... ]
            beta: np.beta(10, ), default to be none
            seqname: s11__xxx__cam_x
        '''
        if seqname is None:
            seqname = 'none'
        if beta is None:
            beta = np.array([0]*10)
            reload_robot = False
        expert = self.smpl2expert(pose, trans, beta, seqname)
        experts = []
        experts.append(expert)
        data = [experts, *self.sample_modules, self.running_state, self.policy_net, self.cc_cfg, self.cfg , self.render, self.custom_reward]
        res = eval_seqs_cmd(self.env, 0, data, reload_robot)
        return res
        
    def res2smpl(self, res, key=None):
        if key is None:
            key = 'none'
        single_res = res[key]
        trans = []
        pose = []
        for i, qpos in enumerate(single_res["pred"]):
            # sim_trans = torch.from_numpy(qpos[:3]).to(torch.float).reshape(-1, 3)
            sim_trans = qpos[:3]
            sim_global_rot = quaternion_to_axis_angle(torch.from_numpy(qpos[3:7].reshape(-1, 4)).to(torch.float32))

            sim_body_pose = torch.from_numpy(qpos[7:]).to(torch.float32).reshape(-1, 3)
            sim_body_pose = euler_angles_to_matrix(sim_body_pose, 'ZYX' )
            sim_body_pose = matrix_to_axis_angle(sim_body_pose)
            sim_pose = torch.cat([sim_global_rot, sim_body_pose])
            smpl2mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
            mujoco2smpl = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]
            # sim_pose_1 = sim_pose[mujoco2smpl].clone().numpy().reshape(-1, 72)
            sim_pose = sim_pose[mujoco2smpl].clone().numpy().reshape(-1)
            pose.append(sim_pose)
            trans.append(sim_trans)
        return pose, trans