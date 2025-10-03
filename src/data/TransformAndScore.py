import sys
import time
from pathlib import Path
from typing import List

import cv2
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, Subset

from data_types.AgenticImage import AgenticImage
from image_aquisition.BasicTestDataset import BasicTestDataset
from juror import Juror
from transformation_agent.TransformationAgent import TransformationAgent
from utils.DataStorage import AgenticImageDataStorage
from utils.Registries import AGENT_FACTORY_REGISTRY, init_registries


class TransformationConfig:
    """
    Configuration class for image transformation and scoring.

    Attributes:
        agents: List of TransformationAgents
    """
    # fixme: Add Sampler to config
    def __init__(self, transformation_agent_factories: List[str], source_dir: str | Path, target_dir: str | Path, sampler: Sampler | None, batch_size: int = 2, num_workers: int = 0):
        init_registries()
        self.agents: List[TransformationAgent] = self.__instantiate_agents(transformation_agent_factories)
        self.source_dir = source_dir if isinstance(source_dir, Path) else Path(source_dir)
        self.target_dir = target_dir if isinstance(target_dir, Path) else Path(target_dir)
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers


    @staticmethod
    def __instantiate_agents(agent_factories: List[str]) -> List:
        agents = []
        for factory_name in agent_factories:
            agents.extend(AGENT_FACTORY_REGISTRY.get(factory_name).create_agents())
        return agents

    def get_agents(self) -> List[TransformationAgent]:
        return self.agents

    def get_source_dir(self) -> Path:
        return self.source_dir

    def get_target_dir(self) -> Path:
        return self.target_dir

    def get_sampler(self) -> Sampler:
        return self.sampler

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_num_workers(self) -> int:
        return self.num_workers




class TransformAndScore:
    """
    This class transforms and scores images provided by Dataloader

    The images are first transformed using the provided transformer factories
    """

    @staticmethod
    def collate_keep_size(batch):
        # Returns a batch without stacking the images, so that they can keep their original size
        return batch

    def __init__(self, config: TransformationConfig):
        self.config = config
        self.dataset: Dataset = BasicTestDataset(root_dir=config.get_source_dir(), max_size=1000)
        sampler = SequentialSampler(Subset(self.dataset, range(20)))
        # Allow multiple workers for batch processing
        self.data_loader: DataLoader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_keep_size,
            sampler=sampler,
            num_workers=config.get_num_workers()  # Use value from config
        )
        self.juror = Juror()
        self.agents: List[TransformationAgent] | None = None


    def transform(self):
        full_start_time = time.perf_counter()
        image_counter = 0
        for batch in self.data_loader:
            batch_start_time = time.perf_counter()
            for image, path, filename in batch:
                image_counter += 1
                processed_image = self.__process_image(image, path, filename)
                AgenticImageDataStorage.save(self.config.get_target_dir(), processed_image)
            batch_end_time = time.perf_counter()
            print(f"Processed batch of {len(batch)} images in {batch_end_time - batch_start_time:.2f} seconds.")

        full_end_time = time.perf_counter()
        print(f"Processed {image_counter} images in {full_end_time - full_start_time:.2f} seconds.")


    def __get_agents(self) -> List[TransformationAgent]:
        if self.agents is None:
            self.agents = self.config.get_agents()
        return self.agents

    def __process_image(self, image, path, filename) -> AgenticImage:
        start_time = time.perf_counter()
        print(f"Image: {filename}: ", end="")
        # preparation and source image scoring
        agentic_image: AgenticImage = AgenticImage()
        agentic_image.update_source_image(image, 'BGR', filename)
        agentic_image.update_source_score(self.__score_image(agentic_image.source_image.get_image_data('RGB')))

        # apply transformations and score them
        agent_count = 0
        total_number_of_agents = len(self.__get_agents())
        for agent in self.__get_agents():
            agent_count += 1
            transformed_image, label = agent.transform(agentic_image.source_image.get_image_data('BGR'))
            score = self.__score_image(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            agentic_image.append_transformer_protocol(label, score)

            # Update agentic image with transformed image, if score is better
            # Prüfe, ob transformed_image ein Objekt mit Attribut score ist
            best_score = getattr(agentic_image.transformed_image, 'score', None)
            if best_score is None or best_score < score:
                agentic_image.update_transformed_image(transformed_image, 'BGR', label, score)

            # Display progress bar, see https://stackoverflow.com/a/15645088
            done = int(50 * agent_count / total_number_of_agents)
            sys.stdout.write(
                f"\r{filename} [{'=' * done}{' ' * (50 - done)}] {(100 * agent_count / total_number_of_agents):.2f} %")
            sys.stdout.flush()

        end_time = time.perf_counter()
        best_score = getattr(agentic_image.transformed_image, 'score', None)
        print(f" -> agents {agent_count} in {end_time - start_time:.2f} seconds. Best image is '{'->'.join(agentic_image.applied_transformers)}' with score {best_score} and change of {agentic_image.calculate_score_change()}")
        return agentic_image





    def __score_image(self, image) -> float:
        score = self.juror.inference(image)
        return score