from pathlib import Path
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parent
XML_PATH = PROJECT_ROOT / "model" / "robotis_ffw" / "scene_ffw_sg2.xml"


def main() -> None:
    """Load only the FFW-SG2 MuJoCo scene and show it in the viewer."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded model: {XML_PATH}")
    print("Close the viewer window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
