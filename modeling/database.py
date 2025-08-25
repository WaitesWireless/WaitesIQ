import torch
from modeling.inference import process_image, get_embeddings
import numpy as np
from PIL import Image
import os
import random
import faiss

class Database:
    def __init__(self, db_path):
        self.waites_path = os.path.join(db_path, "waites")
        self.star_wars_path = os.path.join(db_path, "star_wars")

        self.waites_vector_index = faiss.IndexFlatL2(512)  # Assuming embeddings are of size 512
        self.waites_vector_table = {}

        self.starwars_vector_index = faiss.IndexFlatL2(512)  # Assuming embeddings are of size 512
        self.starwars_vector_table = {}

        self.unknown_vector_index = faiss.IndexFlatL2(512)  # Assuming embeddings are of size 512
        self.unknown_vector_table = {}

        self.compute_waites_embeddings()
        self.compute_star_wars_embeddings()

    def compute_waites_embeddings(self) -> dict[str, torch.Tensor]:
        """
        Compute embeddings for Waites IQ records.
        This function should load the images and compute their embeddings.
        """
        self.waites_embeddings = {}
        names = os.listdir(self.waites_path)
        for name in names:
            path = os.path.join(self.waites_path, name)
            person_embeddings = []
            for image in os.listdir(path):
                image_path = os.path.join(path, image)
                if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(image_path).convert('RGB')
                    boxes, embeddings = process_image(img)
                    if boxes is not None and embeddings is not None:
                        person_embeddings.append(embeddings)
            if len(person_embeddings) > 0:
                embedding = torch.stack(person_embeddings).mean(dim=0).detach().numpy()
                self.waites_vector_table[len(self.waites_vector_table)] = (name, random.randint(160, 190))
                self.waites_vector_index.add(embedding.reshape(1, -1) / np.linalg.norm(embedding))
    
    def compute_star_wars_embeddings(self) -> dict[str, torch.Tensor]:
        """
        Compute embeddings for Star Wars characters.
        This function should load the images and compute their embeddings.
        """
        self.star_wars_embeddings = {}
        star_wars_characters = os.listdir(self.star_wars_path)
        for character in star_wars_characters:
            image_path = os.path.join(self.star_wars_path, character)
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                embeddings = get_embeddings(img_tensor.unsqueeze(0))
                if embeddings is not None:
                    embeddings = embeddings.detach().numpy()
                    self.starwars_vector_table[len(self.starwars_vector_table)] = (character[:-4], random.randint(0, 20))
                    self.starwars_vector_index.add(embeddings.reshape(1, -1) / np.linalg.norm(embeddings))


    @staticmethod
    def _get_uknown_iq() -> int:
        special_seeds = [5, 25, 50, 55, 75]
        unfortunate_seeds = [13]
        seed = random.randint(0, 100)
        if seed in special_seeds:
            return random.randint(140, 155)
        elif seed in unfortunate_seeds:
            return random.randint(20, 40)
        else:
            return random.randint(70, 105)

    def add_unknown(self, embedding: torch.Tensor) -> tuple[str, int]:
        """
        Add an unknown face embedding to the database.
        """
        embedding = embedding.detach().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        D, I = self.unknown_vector_index.search(embedding.reshape(1, -1), k=1)
        if D[0][0] < 0.9:
            return self.unknown_vector_table[I[0][0]]
        else:
            name = f"Unknown_{len(self.unknown_vector_table)}"
            self.unknown_vector_table[len(self.unknown_vector_table)] = (name, self._get_uknown_iq())
            self.unknown_vector_index.add(embedding.reshape(1, -1))
            return self.unknown_vector_table[len(self.unknown_vector_table) - 1]

    def search_waites(self, embedding: torch.Tensor) -> tuple[str, int]:
        """
        Search for Waites IQ records in the database.
        Returns a list of records that match the threshold.
        """
        e = np.expand_dims(embedding.detach().numpy(), axis=0)
        D, I = self.waites_vector_index.search(e / np.linalg.norm(e), k=1)
        distance = D[0][0]
        index = I[0][0]
        if distance < 1.1:
            return self.waites_vector_table[index]
        else:
            return None, None

    def search_star_wars(self, embedding: torch.Tensor) -> str:
        """
        Search for Star Wars characters in the database.
        Returns a list of characters that match the threshold.
        """
        e = np.expand_dims(embedding.detach().numpy(), axis=0)
        _, I = self.starwars_vector_index.search(e / np.linalg.norm(e), k=1)
        index = I[0][0]
        return self.starwars_vector_table[index]


if __name__ == "__main__":
    db = Database("C:/Users/jake.kasper_waites/Work/Repos/WaitesIQ/data")
