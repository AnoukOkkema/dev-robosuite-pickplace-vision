# Import required libraries
from roboflow import Roboflow
from typing import Optional
import yaml
import os
from pathlib import Path

class RoboflowDownloader:
    """
    A class for downloading datasets from Roboflow.

    Attributes:
        api_key (str): The Roboflow API key.
        workspace_name (str): The name of the workspace.
        project_name (str): The name of the project.
        version_number (str): The version number of the dataset.
    """

    def __init__(
        self,
        api_key: str,
        workspace_name: str,
        project_name: str,
        dataset_folder_name: str,
        version_number: str,
        data_path: str,
        model: str,
        logger
    ) -> None:
        """
        Initializes a RoboflowDownloader instance.

        Args:
            api_key (str): The Roboflow API key.
            workspace_name (str): The name of the workspace.
            project_name (str): The Roboflow project slug, used for API calls.
            dataset_folder_name (str): The local folder name to download/read
                the dataset into (independent of the Roboflow project slug,
                since Roboflow's own project.name display name doesn't
                always match what you want on disk).
            version_number (str): The version number of the dataset.

        Returns:
            None
        """
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.project_name = project_name
        self.dataset_folder_name = dataset_folder_name
        self.version_number = version_number
        self.data_path = data_path
        self.model = model
        self.logger = logger

    @staticmethod
    def resolve_local_folder_path(data_path: str, dataset_folder_name: str, version_number) -> str:
        """
        Builds the local dataset folder path from DATASET_FOLDER_NAME +
        VERSION_NUMBER, matching Roboflow's download folder naming
        ("<dataset_folder_name>-<version>") without calling the Roboflow API.

        Used when downloading is disabled but the dataset is already present
        on disk from a previous run.

        Args:
            data_path (str): Root data folder (e.g. config.data_folder).
            dataset_folder_name (str): Local dataset folder name.
            version_number: Dataset version number.

        Returns:
            str: The local project folder path.
        """

        folder_name = f"{dataset_folder_name}-{version_number}"
        return str(Path.cwd() / data_path / folder_name)

    def download_dataset(self, enabled: bool = True) -> Optional[str]:
        """
        Downloads the dataset from Roboflow.

        Args:
            enabled (bool): If False, the download is skipped.

        Returns:
            Optional[str]: The path to the downloaded project folder,
                or None if the download was skipped.
        """

        if not enabled:
            project_folder_path = self.resolve_local_folder_path(
                self.data_path, self.dataset_folder_name, self.version_number
            )

            if os.path.exists(project_folder_path):
                self.logger.info(
                    "Dataset download skipped, using existing local dataset | path=%s",
                    project_folder_path,
                )
                return project_folder_path

            self.logger.warning(
                "Dataset download skipped, but no local dataset found at %s",
                project_folder_path,
            )
            return None

        self.logger.info(
            "Resolving Roboflow dataset | workspace=%s | project=%s | version=%s",
            self.workspace_name,
            self.project_name,
            self.version_number,
        )

        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace_name).project(self.project_name)
        dataset = project.version(self.version_number)
        project_folder_path = self.resolve_local_folder_path(
            self.data_path, self.dataset_folder_name, dataset.version
        )

        # Check if the project folder already exists
        if not os.path.exists(project_folder_path):

            self.logger.info(
                "Dataset not found locally, downloading | model=%s -> %s",
                self.model,
                project_folder_path,
            )

            download = dataset.download(self.model, location=f"{project_folder_path}")

            if download.location is None:
                self.logger.error(
                    "Failed to download dataset. Check your project and version details."
                )
                return None

            project_folder_path = download.location

            self.logger.info(
                "Dataset download completed | path=%s",
                project_folder_path,
            )

        else:
            self.logger.info(
                "Dataset already present locally, skipping download | path=%s",
                project_folder_path,
            )

        # Read the data.yaml file
        with open(f"{project_folder_path}/data.yaml", "r") as file:
            data = yaml.safe_load(file)

        # Update the paths with project_name
        data["train"] = f"{project_folder_path}/train/images"
        data["val"] = f"{project_folder_path}/valid/images"
        data["test"] = f"{project_folder_path}/test/images"

        # Write the updated data back to the file
        with open(f"{project_folder_path}/data.yaml", "w") as file:
            yaml.dump(data, file)

        self.logger.info(
            "Dataset ready | data_yaml=%s/data.yaml",
            project_folder_path,
        )

        return project_folder_path