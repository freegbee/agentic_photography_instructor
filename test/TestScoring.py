import os
from pathlib import Path
from typing import Dict

import cv2
from numpy import ndarray

from data_types.AgenticImage import AgenticImage
from juror import Juror
from utils.ConfigLoader import ConfigLoader
from utils.Registries import AGENT_FACTORY_REGISTRY, init_registries
from utils.TestingUtils import TestingUtils


def main():
    init_registries()
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_path = Path.cwd() / Path(config['dev']['single_image'])
    # Load image in BGR pixel order
    image: ndarray = TestingUtils.load_image_from_path(image_path)

    agentic_image = AgenticImage()
    agentic_image.update_source_image(image, 'BGR', image_path.name)

    juror = Juror()
    score = juror.inference(agentic_image.source_image.get_image_data('RGB'))
    agentic_image.source_image.score = score

    print(f"Score for image {agentic_image.filename}: {score}")

    for factory_name in AGENT_FACTORY_REGISTRY.keys():
        for agent in AGENT_FACTORY_REGISTRY.get(factory_name).create_agents():
            transformed_image, label = agent.transform(agentic_image.source_image.get_image_data('BGR'))
            score = juror.inference(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            agentic_image.append_transformer_protocol(label, score)
            print(".", end="")

            # Update agentic image with transformed image, if score is better
            if agentic_image.transformed_image is None or agentic_image.transformed_image.score is None or agentic_image.transformed_image.score < score:
                agentic_image.update_transformed_image(transformed_image, 'BGR', label, score)

    print()
    print(
        f"Best image is '{'->'.join(agentic_image.applied_transformers)}' with score {agentic_image.transformed_image.score} and change of {agentic_image.calculate_score_change()}")

    print("Full protocol:")
    for protocol in agentic_image.transformer_protocol:
        print(f" - Transformers {'->'.join(protocol.applied_transformers)} resulted in score {protocol.score} with change {protocol.score_change}")


if __name__ == '__main__':
    main()
