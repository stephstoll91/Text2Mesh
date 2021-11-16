from Pose2Mesh.smplifyx.run_inference import run_inference
from Pose2Mesh.smplifyx.calc_bleu import calc_bleu
from Pose2Mesh.smplifyx.cmd_parser import parse_config

if __name__ == "__main__":
    args = parse_config()
    run_inference(**args)
    calc_bleu(**args)
