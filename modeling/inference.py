from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

mtcnn = MTCNN(keep_all=True).eval()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def process_image(image):
    """
    Process an image to extract face embeddings.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        Tensor: The embeddings of the detected face.
    """
    boxes, _ = mtcnn.detect(image)
    
    if boxes is not None:
        faces = mtcnn.extract(image, batch_boxes=boxes, save_path=None)
        embeddings = resnet(faces)
        return boxes, embeddings
    else:
        return None, None

def get_embeddings(face) -> torch.Tensor:
    """
    Get embeddings for a detected face using InceptionResnetV1.
    
    Args:
        face (Tensor): The detected face tensor.
        
    Returns:
        Tensor: The embeddings of the face.
    """
    embeddings = resnet(face)
    return embeddings

def compare_faces(embeddings1, embeddings2, threshold=0.6):
    """
    Compare two face embeddings to determine if they are similar.
    
    Args:
        embeddings1 (Tensor): The first face embedding.
        embeddings2 (Tensor): The second face embedding.
        threshold (float): The similarity threshold.
        
    Returns:
        bool: True if faces are similar, False otherwise.
    """
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)
    return cosine_sim.item() > threshold