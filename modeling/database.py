import torch
from modeling.inference import process_image, get_embeddings
import numpy as np
from PIL import Image
import os
import random

class Database:
    def __init__(self, db_path):
        self.waites_path = os.path.join(db_path, "waites")
        self.star_wars_path = os.path.join(db_path, "star_wars")
        self.compute_waites_embeddings()
        self.compute_star_wars_embeddings()
        for person, (embedding, iq) in self.waites_embeddings.items():
            print(f"Computed embedding for {person} with shape {embedding.shape}")

        for character, embedding in self.star_wars_embeddings.items():
            print(f"Computed embedding for {character} with shape {embedding.shape}")

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
                self.waites_embeddings[name] = (torch.stack(person_embeddings).mean(dim=0), random.randint(0, 200))
    
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
                    self.star_wars_embeddings[character[:-4]] = embeddings
    
    def compare_embedding(self, embedding1, embedding2, threshold=0.6):
        """
        Compare the given embedding with the stored embeddings.
        Returns True if the embedding matches any of the stored embeddings above the threshold.
        """
        return torch.cosine_similarity(embedding1, embedding2).item() > threshold

    def search_waites(self, embedding) -> tuple[str, int]:
        """
        Search for Waites IQ records in the database.
        Returns a list of records that match the threshold.
        """
        for person, (person_embedding, person_iq) in self.waites_embeddings.items():
            if self.compare_embedding(embedding, person_embedding):
                return person, person_iq
        return None, None
    
    def search_star_wars(self, embedding) -> str:
        """
        Search for Star Wars characters in the database.
        Returns a list of characters that match the threshold.
        """
        sims = [
            (character, torch.nn.functional.cosine_similarity(embedding, char_embedding).item())
            for character, char_embedding in self.star_wars_embeddings.items()
        ]
        max_similarity = max(sims, key=lambda x: x[1])
        return max_similarity[0] 

if __name__ == "__main__":
    db = Database("C:/Users/jake.kasper_waites/Work/Repos/WaitesIQ/data")
