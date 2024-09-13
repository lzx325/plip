import os
import glob as glob
import pickle as pkl

import numpy as np
import scipy as sp
import pandas as pd
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

def list_images_nonrecursively(directory):
    image_suffixes = ["jpg", "png","JPEG","PNG"]
    image_files = list()
    for suffix in image_suffixes:
        image_files += glob.glob(os.path.join(directory, f"*.{suffix}"))
    return image_files
    
class DatabaseEmbedder:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.database_embeddings = None
        self.database_image_fp_list = None
        self.metadata = None

    def embed(self, database):
        return self.embed_fn(database)

    def init_from_directory(self, directory, use_cache = True, metadata = None):
        if use_cache and os.path.exists(os.path.join(directory,"database_embeddings.pkl")):
            with open(os.path.join(directory,"database_embeddings.pkl"), "rb") as f:
                database_embeddings = pkl.load(f)
            self.database_embeddings = database_embeddings["database_embeddings"]
            self.database_image_fp_list = database_embeddings["database_image_fp_list"]
            self.metadata = metadata
            
        image_fp_list = list_images_nonrecursively(directory)
        database_embeddings = self.init_from_image_fp_list(image_fp_list, metadata=metadata)
        # save embeddings to disk
        with open(os.path.join(directory,"database_embeddings.pkl"), "wb") as f:
            pkl.dump(
                {
                    "database_embeddings":database_embeddings, 
                    "database_image_fp_list":image_fp_list,
                    "metadata":metadata
                }, f
            )
        return database_embeddings

    def init_from_image_fp_list(self, image_fp_list, metadata = None):
        if metadata is not None:
            assert all([image_fp in metadata.index for image_fp in image_fp_list]), "Metadata should contain all image file names"
            # subset metadata
            self.metadata = metadata.loc[image_fp_list].copy()
        database_embeddings = list()
        for image_fp in image_fp_list:
            image = Image.open(image_fp)
            image = self.embed(image)
            database_embeddings.append(image)
        database_embeddings = np.stack(database_embeddings)
        self.database_embeddings = database_embeddings
        self.database_image_fp_list = image_fp_list
        return database_embeddings
    
    def fp_to_index(self, image_fp):
        if type(image_fp) == str:
            return self.database_image_fp_list.index(image_fp)
        elif type(image_fp) == list:
            return [self.database_image_fp_list.index(fp) for fp in image_fp]

class QueryEmbedder:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.query_embeddings = None
        self.query_image_fp_list = None
        self.metadata = None

    def embed(self, query):
        return self.embed_fn(query)

    def init_from_directory(self, directory, metadata = None):

        image_fp_list = list_images_nonrecursively(directory)
        query_embeddings = self.init_from_image_fp_list(image_fp_list, metadata=metadata)
        return query_embeddings

    def init_from_image_fp_list(self, image_fp_list, metadata = None):
        if metadata is not None:
            assert all([image_fp in metadata.index for image_fp in image_fp_list]), "Metadata should contain all image file names"
            # subset metadata
            self.metadata = metadata.loc[image_fp_list].copy()
        query_embeddings = list()
        for image_fp in image_fp_list:
            image = Image.open(image_fp)
            image = self.embed(image)
            query_embeddings.append(image)
        query_embeddings = np.stack(query_embeddings)
        self.query_embeddings = query_embeddings
        self.query_image_fp_list = image_fp_list
        return query_embeddings
    def fp_to_index(self, image_fp):
        if type(image_fp) == str:
            return self.query_image_fp_list.index(image_fp)
        elif type(image_fp) == list:
            return [self.query_image_fp_list.index(fp) for fp in image_fp]


class ImageRetriever:
    def __init__(self, K=50):
        self.evaluators = {
            "Recall@6" : RecallAtK(6),
            "Recall@10": RecallAtK(10),
            # "Recall@50": RecallAtK(50),
            "NDCG@6": NDCGAtK(6),
            "NDCG@10": NDCGAtK(10),
            # "NDCG@50": NDCGAtK(50)
        }
        self.K = K
    def run_evaluators(self, ground_truth_list, retrieved_indices):
        metrics = dict()
        for evaluator_name, evaluator in self.evaluators.items():
            metrics[evaluator_name] = evaluator(ground_truth_list, retrieved_indices)
        return metrics

    def retrieval(self, database_embeddings, query_embeddings, ground_truth_list = None, K = None):
        if K is None:
            K = self.K
        if ground_truth_list is not None:
            assert len(ground_truth_list) == len(query_embeddings), "Ground truth list and query embeddings should have the same"
        retrieved_indices = list()

        for t in query_embeddings:
            arr = t.dot(database_embeddings.T)

            best = arr.argsort()[-K:][::-1]

            retrieved_indices.append(best)

        if ground_truth_list is not None:
            metrics=self.run_evaluators(ground_truth_list,retrieved_indices)
            return retrieved_indices, metrics
        else:
            return retrieved_indices, None

class RecallAtK:
    def __init__(self, K):
        self.K = K
    def __call__(self, ground_truth_list, retrieved_indices):
        recall = 0
        for i in range(len(ground_truth_list)):
            assert len(ground_truth_list[i]) > 0, "Ground truth list is empty"
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            correct = len(set(ground_truth_list[i]) & set(retrieved_indices[i][:self.K]))
            recall += ( 1 / len(ground_truth_list) ) * correct / len(ground_truth_list[i])
        return recall

class NDCGAtK:
    def __init__(self, K):
        self.K = K
    def __call__(self, ground_truth_list, retrieved_indices):
        ndcg = 0
        for i in range(len(ground_truth_list)):
            assert len(ground_truth_list[i]) > 0, "Ground truth list is empty"
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            dcg = 0
            idcg = 0
            for j in range(self.K):
                if retrieved_indices[i][j] in ground_truth_list[i]:
                    dcg += 1 / np.log2(j + 2)
                if j < len(ground_truth_list[i]):
                    idcg += 1 / np.log2(j + 2)
            ndcg += ( 1 / len(ground_truth_list) ) * dcg / idcg
        return ndcg

class ImageSearcher:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
    @staticmethod
    def parse_metadata(image_fp_list):
        metadata = {"image_class": list()}
        for image_fp in image_fp_list:
            metadata["image_class"].append(os.path.basename(image_fp).split("_")[0])
        metadata = pd.DataFrame(metadata, index=image_fp_list)
        return metadata

    @staticmethod
    def get_ground_truth_list(database_eb, query_eb):
        ground_truth_list=list()
        for query_fp in query_eb.query_image_fp_list:
            gt_index = database_eb.metadata.index[database_eb.metadata["image_class"] == query_eb.metadata.loc[query_fp]["image_class"]].tolist()
            gt_index = database_eb.fp_to_index(gt_index)
            ground_truth_list.append(gt_index)

        return ground_truth_list

    @staticmethod
    def get_ground_truth_list2(database_eb, query_eb):
        ground_truth_list=list()
        for query_fp in query_eb.query_image_fp_list:
            gt_index = database_eb.fp_to_index(query_fp)
            ground_truth_list.append([gt_index])
        return ground_truth_list

    def search(self, database_dir, query_image_dir):
        database_image_fp_list = list_images_nonrecursively(database_dir)
        database_image_metadata = self.parse_metadata(database_image_fp_list)
        database_eb = DatabaseEmbedder(self.embed_fn)
        database_eb.init_from_directory(database_dir, metadata=database_image_metadata)

        query_image_fp_list = list_images_nonrecursively(query_image_dir)
        query_image_metadata = self.parse_metadata(query_image_fp_list)
        query_eb = QueryEmbedder(self.embed_fn)
        query_eb.init_from_directory(query_image_dir, metadata=query_image_metadata)
        
        ground_truth_list = self.get_ground_truth_list(database_eb, query_eb)
        ir = ImageRetriever(K=10)
        return ir.retrieval(database_eb.database_embeddings, query_eb.query_embeddings, ground_truth_list)

class DefaultEmbedding:
    def __call__(self, image):
        emb = np.array(image).mean(axis=(0,1))
        emb /= np.linalg.norm(emb)
        return emb

class PLIPEmbedding:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("downloaded_data/plip")
        self.processor = CLIPProcessor.from_pretrained("downloaded_data/plip")
    def __call__(self, image):
        inputs = self.processor(text=["a histopathological image"],images=[image], return_tensors="pt")
        outputs = self.model(**inputs)
        emb = outputs["vision_model_output"].pooler_output[0].detach().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb

if __name__=="__main__":
    np.random.seed(0)
    database_dir="mutation_images/image_search_examples/database_images"
    query_image_dir="mutation_images/image_search_examples/query_images"
    searcher = ImageSearcher(PLIPEmbedding())
    retrieved_indices, metrics = searcher.search(database_dir, query_image_dir)
    print(metrics)
    