import numpy as np
import joblib
import cv2
import glfw
import glob
import os
import sys
import pdb
import os.path as osp
import shutil
import time
import subprocess

sys.path.append(os.getcwd())
sys.path.append('phys_new')
from uhc.utils.image_utils import write_frames_to_video
from uhc.khrylib.rl.utils.visualizer import Visualizer
from uhc.utils.transformation import quaternion_twovec
import mujoco_py


class CopycatVisualizer(Visualizer):

    def __init__(self, vis_file, agent, res=None):
        self.agent = agent
        super().__init__(vis_file)

        # self.env_vis.model.geom_rgba[1:ngeom + 2 ]= np.array([0.7, 0.0, 0.0, 0])

        self.T = 12
        self.image_acc = []
        
        if hasattr(agent.cfg,'old'):
            self.data_gen = self.data_generator_old()
        else :
            if res is not None:
                self.data_gen = self.data_generator(res)
            else:
                self.data_gen = self.data_generator()
        self.data = next(self.data_gen)

        self.setup_viewing_angle()

    def set_video_path(self, image_path, video_path):
        self.env_vis.set_video_path(image_path=image_path, video_path=video_path)

    def setup_viewing_angle(self):
        # ngeom_new = len(self.agent.env.converter.new_joint_names)
        # ngeom_old = len(self.agent.env.converter.smpl_joint_names)
        # self.env_vis.model.geom_rgba[ngeom_new + 11 :] = np.array([0.7, 0.0, 0.0, 1])

        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = 45
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.env_vis.set_custom_key_callback(self.key_callback)

    def data_generator(self, res=None):

        if self.agent.cfg.mode != "disp_stats":
            if res is not None:
                for i,key in enumerate(res):
                        self.cur_key = key
                        eval_res = res[key]
                        print(f"Generating for {key} seqlen: {len(eval_res['gt'])}")

                        print(
                            "Agent Mass:",
                            mujoco_py.functions.mj_getTotalmass(self.agent.env.model),
                        )
                        print_str = "\t".join([f"{k}: {v:.3f}" for k, v in eval_res.items() if not k in [
                            "gt",
                            "pred",
                            "pred_jpos",
                            "gt_jpos",
                            "reward",
                            "gt_vertices",
                            "pred_vertices",
                            "gt_joints",
                            "pred_joints",
                            "action",
                            "vf_world",
                        ] and (not isinstance(v, np.ndarray))])
                        self.env_vis.reload_sim_model(
                            self.agent.env.smpl_robot.export_vis_string(num_cones=int(self.agent.env.vf_dim / self.agent.env.body_vf_dim) if self.agent.cc_cfg.residual_force and self.agent.cfg.render_rfc and self.agent.cc_cfg.residual_force_mode == "explicit" else 0,).decode("utf-8"))

                        self.setup_viewing_angle()
                        # self.set_video_path(
                        #     image_path=osp.join(
                        #         self.agent.cfg.output,
                        #         take_key,
                        #         f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
                        #     ),
                        #     video_path=osp.join(
                        #         self.agent.cfg.output,
                        #         f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
                        #     ),
                        # )
                        # os.makedirs(osp.join(self.agent.cfg.output, take_key), exist_ok=True)
                        self.num_fr = eval_res["pred"].shape[0]
                        yield eval_res

            else :
                for task in self.agent.tasks:
                    data = [task,*self.agent.sample_modules,self.agent.running_state,self.agent.policy_net,self.agent.cc_cfg,self.agent.cfg ,self.agent.render,self.agent.custom_reward]
                    from uhc.agents.agent_copycat_zc import eval_seqs_cmd
                    res = eval_seqs_cmd(self.agent.env,0,data)
                    for i,key in enumerate(res):
                        self.cur_key = key
                        eval_res = res[key]
                        print(f"Generating for {key} seqlen: {len(eval_res['gt'])}")

                        print(
                            "Agent Mass:",
                            mujoco_py.functions.mj_getTotalmass(self.agent.env.model),
                        )
                        print_str = "\t".join([f"{k}: {v:.3f}" for k, v in eval_res.items() if not k in [
                            "gt",
                            "pred",
                            "pred_jpos",
                            "gt_jpos",
                            "reward",
                            "gt_vertices",
                            "pred_vertices",
                            "gt_joints",
                            "pred_joints",
                            "action",
                            "vf_world",
                        ] and (not isinstance(v, np.ndarray))])
                        self.env_vis.reload_sim_model(
                            self.agent.env.smpl_robot.export_vis_string(num_cones=int(self.agent.env.vf_dim / self.agent.env.body_vf_dim) if self.agent.cc_cfg.residual_force and self.agent.cfg.render_rfc and self.agent.cc_cfg.residual_force_mode == "explicit" else 0,).decode("utf-8"))

                        self.setup_viewing_angle()
                        # self.set_video_path(
                        #     image_path=osp.join(
                        #         self.agent.cfg.output,
                        #         take_key,
                        #         f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
                        #     ),
                        #     video_path=osp.join(
                        #         self.agent.cfg.output,
                        #         f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
                        #     ),
                        # )
                        # os.makedirs(osp.join(self.agent.cfg.output, take_key), exist_ok=True)
                        self.num_fr = eval_res["pred"].shape[0]
                        yield eval_res
        else:
            yield None


    def data_generator_old(self):

        if self.agent.cfg.mode != "disp_stats":
            for loader in self.agent.test_data_loaders:
                for take_key in loader.data_keys:

                    self.cur_key = take_key
                    print(f"Generating for {take_key} seqlen: {loader.get_sample_len_from_key(take_key)}")
                    eval_res = self.agent.eval_seq(take_key, loader) 
                    # data = [loader.data_keys, loader,*self.agent.sample_modules,self.agent.running_state,self.agent.policy_net,self.agent.cc_cfg,self.agent.cfg ,self.agent.render,self.agent.custom_reward]
                    # from uhc.agents.agent_copycat import eval_seqs_cmd
                    # res = eval_seqs_cmd(self.agent.env,0,data)
                    # eval_res = res[take_key]
                    print(
                        "Agent Mass:",
                        mujoco_py.functions.mj_getTotalmass(self.agent.env.model),
                    )
                    print_str = "\t".join([f"{k}: {v:.3f}" for k, v in eval_res.items() if not k in [
                        "gt",
                        "pred",
                        "pred_jpos",
                        "gt_jpos",
                        "reward",
                        "gt_vertices",
                        "pred_vertices",
                        "gt_joints",
                        "pred_joints",
                        "action",
                        "vf_world",
                    ] and (not isinstance(v, np.ndarray))])
                    self.env_vis.reload_sim_model(
                        self.agent.env.smpl_robot.export_vis_string(num_cones=int(self.agent.env.vf_dim / self.agent.env.body_vf_dim) if self.agent.cc_cfg.residual_force and self.agent.cfg.render_rfc and self.agent.cc_cfg.residual_force_mode == "explicit" else 0,).decode("utf-8"))

                    self.setup_viewing_angle()
                    self.set_video_path(
                        image_path=osp.join(
                            self.agent.cfg.output,
                            take_key,
                            f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
                        ),
                        video_path=osp.join(
                            self.agent.cfg.output,
                            f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
                        ),
                    )
                    os.makedirs(osp.join(self.agent.cfg.output, take_key), exist_ok=True)
                    self.num_fr = eval_res["pred"].shape[0]
                    yield eval_res
        else:
            yield None

    def get_cylinder_attr(self, x1, x2, thickness=0.008):
        pos = (x1 + x2) * 0.5
        v1 = np.array([0, 0, 1])
        v2 = x2 - x1
        scale = np.linalg.norm(v2)
        v2 /= scale
        quat = quaternion_twovec(v1, v2)
        size = np.array([thickness, scale / 2, 0])
        return pos, quat, size

    def render_contact_force(self, i, contact_point, force):
        # contact_point[2] -= 0.03
        pos, quat, size = self.get_cylinder_attr(contact_point, contact_point + force)
        self.env_vis.model.geom_pos[i + 1] = pos
        self.env_vis.model.geom_quat[i + 1] = quat.copy()

        self.env_vis.model.geom_size[i + 1] = size
        self.env_vis.model.geom_rgba[i + 1] = np.array([0.0, 0.8, 1.0, 1.0])

        self.env_vis.model.body_pos[i + len(self.env_vis.model.body_names)] = (contact_point + force)
        self.env_vis.model.body_quat[i + len(self.env_vis.model.body_names)] = quat
        # self.env_vis.model.geom_rgba[ i+ 49] = np.array([0.0, 0.8, 1.0, 1.0])

    def render_virtual_force(self, vf):
        # for i in range(20):
        #     self.env_vis.model.geom_rgba[i + 1, 3] = 0.0
        #     self.env_vis.model.geom_rgba[-20 + i, 3] = 0.0
        vf = vf.reshape(-1, self.agent.env.body_vf_dim)

        for i, x in enumerate(vf):
            # if np.linalg.norm(x) < 1e-3:
            #     continue
            contact_point = x[:3].copy()
            force = x[3:6] * 2
            self.render_contact_force(i, contact_point, force)

    def update_pose(self):

        # if self.env_vis.viewer._record_video:
        #     print(self.fr)
        # print(self.fr)
        expert = self.agent.env.expert
        lim = self.agent.env.converter.new_nq + (expert["obj_pose"].shape[1] if expert["has_obj"] else 0)

        # self.data["pred"][self.fr][-14:] = expert["obj_pose"][self.fr]

        self.env_vis.data.qpos[:lim] = self.data["pred"][self.fr]
        self.env_vis.data.qpos[lim:] = self.data["gt"][self.fr]

        if (self.agent.cfg.render_rfc and self.agent.cc_cfg.residual_force and self.agent.cc_cfg.residual_force_mode == "explicit"):

            self.render_virtual_force(self.data["vf_world"][self.fr])

        # self.env_vis.data.qpos[env.model.nq] += 1.0
        # if args.record_expert:
        # self.env_vis.data.qpos[:env.model.nq] = self.data['gt'][self.fr]
        if self.agent.cfg.hide_im:
            self.env_vis.data.qpos[2] = 100.0

        if self.agent.cfg.hide_expert:
            self.env_vis.data.qpos[lim + 2] = 100.0
        if self.agent.cfg.shift_expert:
            self.env_vis.data.qpos[lim] += 1

        if self.agent.cfg.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
        self.env_vis.sim_forward()
        # print(self.fr)
        if self.agent.cfg.render_video:
            size = (1920, 1080)
            data = np.asarray(
                self.env_vis.viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :],
                dtype=np.uint8,
            )
            self.image_acc.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

    def key_callback(self, key, action, mods):

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            self.T = self.T_arr[(self.T_arr.index(self.T) + 1) % len(self.T_arr)]
            print(f"T: {self.T}")
        elif key == glfw.KEY_F:
            self.T = self.T_arr[(self.T_arr.index(self.T) - 1) % len(self.T_arr)]
            print(f"T: {self.T}")
        elif key == glfw.KEY_Q:
            self.data = next(self.data_gen, None)
            if self.data is None:
                print("end of data!!")
                exit()
            self.fr = 0
            self.update_pose()
        elif key == glfw.KEY_W:
            self.fr = 0
            self.update_pose()
        elif key == glfw.KEY_E:
            self.fr = self.num_fr - 1
            self.update_pose()
        elif key == glfw.KEY_G:
            self.repeat = not self.repeat
            self.update_pose()
        elif key == glfw.KEY_S:
            self.reverse = not self.reverse
        elif key == glfw.KEY_RIGHT:
            if self.fr < self.num_fr - 1:
                self.fr += 1
            self.update_pose()
        elif key == glfw.KEY_LEFT:
            if self.fr > 0:
                self.fr -= 1
            self.update_pose()

        elif key == glfw.KEY_B:
            self.agent.cfg.hide_expert = not self.agent.cfg.hide_expert
        elif key == glfw.KEY_M:
            self.agent.cfg.hide_im = not self.agent.cfg.hide_im

        elif key == glfw.KEY_N:
            self.agent.cfg.shift_expert = not self.agent.cfg.shift_expert
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        else:
            return False
        return True

    def display_coverage(self):
        res_dir = osp.join(
            self.agent.cfg.output_dir,
            f"{self.agent.cfg.epoch}_{self.agent.data_loader.name}_coverage_full.pkl",
        )
        print(res_dir)
        data_res = joblib.load(res_dir)
        print(len(data_res))

        def vis_gen():
            keys = sorted(list(data_res.keys()))
            keys = list(data_res.keys())
            # keys = sorted(
            #     [
            #         k
            #         for k in list(data_res.keys())
            #         if data_res[k]["percent"] != 1
            #         or ("fail_safe" in data_res[k] and data_res[k]["fail_safe"])
            #     ]
            # )
            for k in keys:
                v = data_res[k]
                print(f"{v['percent']:.3f} |  {k}")
                self.num_fr = len(v["pred"])
                yield v

        self.data_gen = iter(vis_gen())
        self.data = next(self.data_gen)
        self.show_animation()
    
    def record_video(self, preview=False, batch_id=None):
        frame_dir = f"{self.agent.cfg.output_dir}/frames"
        if batch_id is not None:
            frame_dir = frame_dir + '_%03d' %batch_id
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            for _ in range(20):
                self.render()
            if not preview:
                t0 = time.time()
                save_screen_shots(
                    self.env_vis.viewer.window, f"{frame_dir}/%04d.png" % fr, autogui=True
                )
                print("%d/%d, %.3f" % (fr, self.num_fr, time.time() - t0))

        if not preview:
            out_name = f'{self.agent.cfg.output_dir}/video_{batch_id}.mp4'
            # cmd = [
            #     "/usr/local/bin/ffmpeg",
            #     "-y",
            #     "-r",
            #     "30",
            #     "-f",
            #     "image2",
            #     "-start_number",
            #     "0",
            #     "-i",
            #     f"{frame_dir}/%04d.png",
            #     "-vcodec",
            #     "libx264",
            #     "-crf",
            #     "5",
            #     "-pix_fmt",
            #     "yuv420p",
            #     out_name,
            # ]
            cmd = [
                "ffmpeg",
                "-y",
                "-r",
                "30",
                "-f",
                "image2",
                "-start_number",
                "0",
                "-i",
                f"{frame_dir}/%04d.png",
                "-vcodec",
                "libx264",
                "-crf",
                "5",
                "-pix_fmt",
                "yuv420p",
                out_name,
            ]
            subprocess.call(cmd)


def save_screen_shots(window, file_name, transparent=False, autogui=False):
    import glfw
    xpos, ypos = glfw.get_window_pos(window)
    width, height = glfw.get_window_size(window)
    if autogui:
        import pyautogui
        # image = pyautogui.screenshot(region=(xpos*2, ypos*2, width*2, height*2))
        image = pyautogui.screenshot(region=((xpos//2)*2, (ypos//2)*2, (width//2)*2, (height//2)*2))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA if transparent else cv2.COLOR_RGB2BGR)
        if transparent:
            image[np.all(image >= [240, 240, 240, 240], axis=2)] = [255, 255, 255, 0]
        cv2.imwrite(file_name, image)
    else:
        print(width*2, height*2)
        subprocess.call(['screencapture', '-x', '-m', f'-R {xpos},{ypos},{width},{height}', file_name])
