import mujoco
import mujoco.viewer
import time
import math

class JudoSim:
    def __init__(self, model_xml='judo_athlete.xml'):
        print(f"Loading MuJoCo Model: {model_xml}")
        # 1. Load the MJCF model and create the data state
        try:
            self.model = mujoco.MjModel.from_xml_path(model_xml)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"Failed to load XML. Ensure {model_xml} is in the same folder.")
            raise e

    def run_live_viewer(self):
        """Opens the interactive MuJoCo window with a test animation."""
        print("Starting interactive viewer. Press ESC to close.")
        
        # 2. Launch the passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # Find the ID of our right knee motor so we can send it commands
            knee_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right_knee")
            
            start_time = time.time()
            
            while viewer.is_running():
                step_start = time.time()
                elapsed = step_start - start_time
                
                # --- TEST MOTOR INJECTION ---
                # We use a sine wave to bend the knee between 0 and -90 degrees
                # In the final version, this is where your Stage 6 JSON data goes!
                target_angle = (math.sin(elapsed * 2) - 1) * 45  
                
                # Convert to radians for MuJoCo (degrees * pi/180)
                target_rads = target_angle * (math.pi / 180.0)
                self.data.ctrl[knee_motor_id] = target_rads
                # ----------------------------

                # 3. Step the physics simulation forward by 1 frame
                mujoco.mj_step(self.model, self.data)
                
                # 4. Update the visual window
                viewer.sync()
                
                # 5. Timing Control (Keep it at 60fps / 0.0166s per step)
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    sim = JudoSim()
    sim.run_live_viewer()