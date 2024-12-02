import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import glob as glob
import pickle as pkl

import numpy as np
import scipy as sp
import pandas as pd
from PIL import Image

import torch as th
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from guided_diffusion.filename_utils import parse_directory_filenames

def list_images_nonrecursively(directory):
    image_suffixes = ["jpg", "png","JPEG","PNG"]
    image_files = list()
    for suffix in image_suffixes:
        image_files += glob.glob(os.path.join(directory, f"*.{suffix}"))
    return sorted(image_files)
    
class DatabaseEmbedder:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.database_embeddings = None
        self.database_image_fp_list = None
        self.metadata = None

    def embed(self, database):
        return self.embed_fn(database)

    def init_from_directory(self, directory, use_cache = True, metadata = None):
        embedding_fn = "database_embeddings.{}.pkl".format(self.embed_fn.__class__.__name__)
        embedding_fp = os.path.join(directory, embedding_fn)
        if use_cache and os.path.exists(embedding_fp):
            with open(embedding_fp, "rb") as f:
                database_embeddings = pkl.load(f)
            self.database_embeddings = database_embeddings["database_embeddings"]
            self.database_image_fp_list = database_embeddings["database_image_fp_list"]
            self.metadata = metadata
            return database_embeddings
            
        image_fp_list = list_images_nonrecursively(directory)
        database_embeddings = self.init_from_image_fp_list(image_fp_list, metadata=metadata)
        # save embeddings to disk
        with open(embedding_fp, "wb") as f:
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
        for image_fp in tqdm(image_fp_list):
            image = Image.open(image_fp).convert("RGB")
            image = self.embed(image)
            database_embeddings.append(image)
        database_embeddings = np.concatenate(database_embeddings,axis=0)
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
        for image_fp in tqdm(image_fp_list):
            image = Image.open(image_fp).convert("RGB")
            image = self.embed(image)
            query_embeddings.append(image)
        query_embeddings = np.concatenate(query_embeddings,axis=0)
        self.query_embeddings = query_embeddings
        self.query_image_fp_list = image_fp_list
        return query_embeddings
    def fp_to_index(self, image_fp):
        if type(image_fp) == str:
            return self.query_image_fp_list.index(image_fp)
        elif type(image_fp) == list:
            return [self.query_image_fp_list.index(fp) for fp in image_fp]


class ImageRetriever:
    def __init__(self, K=50, evaluators = None):
        if evaluators is None:
            self.evaluators = {
                "Precision@3": PrecisionAtK(3),
                "Precision@3PerItem": PrecisionAtKPerItem(3),
                "Precision@5": PrecisionAtK(5),
                "Precision@10": PrecisionAtK(10),
                "Precision@50": PrecisionAtK(50),
                "Recall@5": RecallAtK(5),
                "Recall@10": RecallAtK(10),
                "Recall@50": RecallAtK(50),
                "NDCG@5": NDCGAtK(5),
                "NDCG@10": NDCGAtK(10),
                "NDCG@50": NDCGAtK(50),
                "CosineDistance@3": CosineDistanceAtK(3, only_correct=True),
                "CosineDistance@5": CosineDistanceAtK(5, only_correct=True)
            }
        self.K = K

    def run_evaluators(self, ground_truth_list, retrieved_indices, retrieved_scores):
        metrics = dict()
        for evaluator_name, evaluator in self.evaluators.items():
            metrics[evaluator_name] = evaluator(
                ground_truth_list = ground_truth_list, 
                retrieved_indices = retrieved_indices, 
                retrieved_scores = retrieved_scores
            )
        return metrics

    def retrieval(self, database_embeddings, query_embeddings, ground_truth_list = None, K = None):
        if K is None:
            K = self.K
        if ground_truth_list is not None:
            assert len(ground_truth_list) == len(query_embeddings), "Ground truth list and query embeddings should have the same"
        retrieved_indices = list()
        retrieved_scores = list()
        for t in query_embeddings:
            if True:
                arr = t.dot(database_embeddings.T)
                best = arr.argsort()[-K:][::-1]
                score = arr[best]

            if False:
                # random select K indices
                arr = t.dot(database_embeddings.T)
                best = np.random.choice(len(database_embeddings), K, replace=False)
                score = arr[best]

            retrieved_indices.append(best)
            retrieved_scores.append(score)

        if ground_truth_list is not None:
            metrics=self.run_evaluators(
                ground_truth_list = ground_truth_list, 
                retrieved_indices = retrieved_indices, 
                retrieved_scores = retrieved_scores
            )
            return retrieved_indices, retrieved_scores, metrics
        else:
            return retrieved_indices, retrieved_scores, None
        
class PrecisionAtK:
    def __init__(self, K):
        self.K = K
    def __call__(self, ground_truth_list, retrieved_indices, **kwargs):
        precision = 0
        for i in range(len(ground_truth_list)):
            assert len(ground_truth_list[i]) > 0, "Ground truth list is empty"
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            correct = len(set(ground_truth_list[i]) & set(retrieved_indices[i][:self.K]))
            if self.K == 10 and correct>=3:
                # print(str(i)+",")
                pass
            precision += ( 1 / len(ground_truth_list) ) * correct / self.K
        return precision

class CosineDistanceAtK:
    def __init__(self, K, only_correct=True):
        self.K = K
        self.only_correct = only_correct
    def __call__(self, ground_truth_list, retrieved_indices, retrieved_scores, **kwargs):
        from collections import defaultdict
        cosd = 0
        result_dict = defaultdict(list)
        for i in range(len(ground_truth_list)):
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            if self.only_correct:
                assert len(ground_truth_list[i]) > 0, "Ground truth list is empty"
                candidate_indices = [j for j in range(self.K) if retrieved_indices[i][j] in ground_truth_list[i]]
                candidate = list(retrieved_indices[i][candidate_indices])
            else:
                candidate = list(retrieved_indices[i][:self.K])
                candidate_indices = list(range(self.K))
            if len(candidate) == 0:
                cosd_current = 1
                cosd_current_arr = np.array([])
            else:
                # convert cosine similarity to cosine distance
                cosd_current_arr = (1 - retrieved_scores[i][candidate_indices])/2
                cosd_current = cosd_current_arr.mean()
            cosd += ( 1 / len(ground_truth_list) ) * cosd_current
            result_dict["cosd"].append(cosd_current)
            result_dict["candidates"].append(candidate)
            result_dict["candidates_cosd"].append(cosd_current_arr.tolist())

        return cosd, pd.DataFrame(result_dict)

class PrecisionAtKPerItem:
    def __init__(self, K):
        self.K = K
    def __call__(self, ground_truth_list, retrieved_indices, **kwargs):
        from collections import defaultdict
        precision = 0
        result_list = defaultdict(list)
        for i in range(len(ground_truth_list)):
            assert len(ground_truth_list[i]) > 0, "Ground truth list is empty"
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            correct = len(set(ground_truth_list[i]) & set(retrieved_indices[i][:self.K]))
            result_list["precision"].append(correct / self.K)
        return pd.DataFrame(result_list)
    
class RecallAtK:
    def __init__(self, K):
        self.K = K
    def __call__(self, ground_truth_list, retrieved_indices, **kwargs):
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
    def __call__(self, ground_truth_list, retrieved_indices, **kwargs):
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

# accuracy of majority voting
class MVAtK:
    def __init__(self, K):
        self.K = K
    def __call__(self, retrieved_indices, database_metadata, query_metadata, **kwargs):
        from collections import defaultdict
        mv = 0
        for i in range(len(retrieved_indices)):
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            prediction = database_metadata.iloc[retrieved_indices[i][:self.K]].mode().iloc[0]
            gt = query_metadata.iloc[i]
            mv += ( 1 / len(retrieved_indices) ) * (gt == prediction)
        return mv
    
class MVAtKPredictions:
    def __init__(self, K):
        self.K = K
    def __call__(self, retrieved_indices, database_metadata, query_metadata, **kwargs):
        from collections import defaultdict
        predictions = defaultdict(list)
        for i in range(len(retrieved_indices)):
            assert len(retrieved_indices[i]) >= self.K, "Retrieved indices list is smaller than K"
            prediction = database_metadata.iloc[retrieved_indices[i][:self.K]].mode().iloc[0]
            gt = query_metadata.iloc[i]
            predictions["prediction"].append(prediction)
            predictions["gt"].append(gt)
        return pd.DataFrame(predictions)
    
class ImageSearcher:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.classification_metrics = {
            "MV@1": MVAtK(1),
            "MV@3": MVAtK(3),
            "MV@5": MVAtK(5),
            "MV@3Predictions": MVAtKPredictions(3),
        }
    @staticmethod
    def parse_metadata(image_fp_list):
        metadata = {"image_class": list()}
        for image_fp in image_fp_list:
            metadata["image_class"].append(os.path.basename(image_fp).split("__")[0])
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

    def search(self, database_dir, query_image_dir, query_image_metadata=None, use_cache = True):
        database_image_fp_list = list_images_nonrecursively(database_dir)
        database_image_fp_list=database_image_fp_list
        database_image_metadata = self.parse_metadata(database_image_fp_list)
        database_eb = DatabaseEmbedder(self.embed_fn)
        database_eb.init_from_directory(database_dir, use_cache=use_cache, metadata=database_image_metadata)

        if query_image_metadata is None:
            query_image_fp_list = list_images_nonrecursively(query_image_dir)
            query_image_metadata = self.parse_metadata(query_image_fp_list)
        else:
            query_image_fp_list = query_image_metadata.index.tolist()


        if False:
            df = pd.read_csv("mutation_images/frozen_translation_search/precision_per_item_df.csv", index_col=0)
            indices = (df["precision"]>=0.3).values.nonzero()[0]
            query_image_fp_list = [query_image_fp_list[i] for i in indices]

        if False:
            df = pd.read_csv("mutation_images/frozen_translation_search/classification_prediction_df.csv",index_col = 0)
            indices = (df["correct"]==True).values.nonzero()[0]
            query_image_fp_list = [query_image_fp_list[i] for i in indices]

        
        
        
        query_eb = QueryEmbedder(self.embed_fn)
        query_eb.init_from_image_fp_list(query_image_fp_list, metadata=query_image_metadata)
        
        ground_truth_list = self.get_ground_truth_list(database_eb, query_eb)
        ir = ImageRetriever(K=50)
        retrieved_indices, retrieved_scores, retrieval_metrics = ir.retrieval(database_eb.database_embeddings, query_eb.query_embeddings, ground_truth_list)
        # add another metric to retrieve the classification results
        for metric_name, metric in self.classification_metrics.items():
            retrieval_metrics[metric_name] = metric(retrieved_indices, database_image_metadata["image_class"], query_image_metadata["image_class"], retrieved_scores = retrieved_scores)
        precision_per_item_df = retrieval_metrics.pop("Precision@3PerItem")
        classification_prediction_df = retrieval_metrics.pop("MV@3Predictions")
        retrieval_metrics.pop("CosineDistance@5")
        cd_score, cd_per_item_df  = retrieval_metrics.pop("CosineDistance@3")
        precision_per_item_df.index = query_image_metadata.index
        classification_prediction_df.index = query_image_metadata.index
        cd_per_item_df.index = query_image_metadata.index
        return retrieved_indices, retrieval_metrics, (precision_per_item_df, classification_prediction_df,cd_per_item_df)

class CLIPClassificationMetadata:
    CLASSIFICATION_TEXTS = {
        "Bladder": "a bladder histopathological image",
        "Breast": "a breast histopathological image",
        "Colorectal": "a colorectal histopathological image",
        "Esphagogastric": "an esophageal or gastric histopathological image",
        "Head Neck": "a head and neck histopathological image",
        "Renal": "a renal histopathological image",
        "Liver": "a liver histopathological image",
        "Lung": "a lung histopathological image",
        "Ovarian": "an ovarian histopathological image",
        "Pancreatic": "a pancreatic histopathological image",
        "Prostate": "a prostate histopathological image",
        "Sarcoma": "a sarcoma histopathological image",
        "Thyroid": "a thyroid histopathological image",
        "Endometrial": "an endometrial histopathological image"
    }

    CLASSIFICATION_TEXTS_2={
        "adipose": "an adipose tissue histopathological image",
        "background": "a background histopathological image",
        "lymphocyte": "a lymphocyte histopathological image",
        "epithelium": "an epithelium histopathological image",
        "mucus": "a mucus histopathological image",
    }

    KIDNEY=["KIRP","KICH","KIRC"]
    GI=["ESCA","STAD","COAD","READ","PAAD"]
    GYN=["OV","BRCA","UCEC"]
    LUNG=["LUAD","LUSC"]
    OTHER=["LIHC","HNSC","SARC","THCA","BLCA","PRAD"]

    TUMOR_FAMILY = {
        "Bladder": "OTHER",
        "Breast": "GYN",
        "Colorectal": "GI",
        "Esphagogastric": "GI",
        "Head Neck": "OTHER",
        "Renal": "KIDNEY",
        "Liver": "OTHER",
        "Lung": "LUNG",
        "Ovarian": "GYN",
        "Pancreatic": "GI",
        "Prostate": "OTHER",
        "Sarcoma": "OTHER",
        "Thyroid": "OTHER",
        "Endometrial": "GYN"
    }

    @staticmethod
    def int2class(int_class):
        return list(CLIPClassificationMetadata.CLASSIFICATION_TEXTS.keys())[int_class]
    @staticmethod
    def class2int(class_name):
        return list(CLIPClassificationMetadata.CLASSIFICATION_TEXTS.keys()).index(class_name)
    @staticmethod
    def class2text(class_name):
        return CLIPClassificationMetadata.CLASSIFICATION_TEXTS[class_name]

class CLIPClassifier:
    def __init__(self, text_embedder, image_embedder):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        # define classification metrics
        self.classification_metrics = {
            "Accuracy": MVAtK(1),
            "Prediction": MVAtKPredictions(1),
        }
        self.text_embeddings, self.text_metadata = self.get_text_embeddings_and_metadata()

    def get_text_embeddings_and_metadata(self):
        text = list(CLIPClassificationMetadata.CLASSIFICATION_TEXTS_2.values())
        text_embeddings = self.text_embedder(text)
        text_metadata = pd.DataFrame({
            "image_class": list(CLIPClassificationMetadata.CLASSIFICATION_TEXTS_2.keys())
        })
        return text_embeddings, text_metadata
    
    # TODO: implementation of classification
    def classify(self, query_image_dir):
        if False:
            query_image_fp_list = list_images_nonrecursively(query_image_dir)
            # random select 1024 images
            np.random.seed(0)
            query_image_fp_list = np.random.choice(query_image_fp_list, 1024, replace=False)
            query_image_metadata = ImageSearcher.parse_metadata(query_image_fp_list)

        if True:
            query_image_metadata = pd.read_csv(
                os.path.join(
                    os.path.dirname(query_image_dir),
                    os.path.basename(query_image_dir).split("_")[0]+"_content_class_classification.csv"
                ),
                index_col=0
            )

            query_image_metadata.index = [os.path.join(query_image_dir, fn) for fn in query_image_metadata.index]
            query_image_fp_list = query_image_metadata.index.tolist()

        query_eb = QueryEmbedder(self.image_embedder)
        query_eb.init_from_image_fp_list(query_image_fp_list, metadata=query_image_metadata)

        ir = ImageRetriever(K=1)
        retrieved_indices, _ = ir.retrieval(self.text_embeddings, query_eb.query_embeddings)
        retrieval_metrics = dict()
        for metric_name, metric in self.classification_metrics.items():
            retrieval_metrics[metric_name] = metric(retrieved_indices, self.text_metadata["image_class"], query_image_metadata["content_class"])
        classification_prediction_df = retrieval_metrics.pop("Prediction")
        classification_prediction_df.index = query_image_metadata.index
        return retrieved_indices, retrieval_metrics, classification_prediction_df

class DefaultEmbedding:
    def __call__(self, image):
        emb = np.array(image).mean(axis=(0,1)).reshape(1,-1)
        emb /= np.linalg.norm(emb)
        return emb

class PLIPEmbedding:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("downloaded_data/plip")
        self.processor = CLIPProcessor.from_pretrained("downloaded_data/plip")
    def __call__(self, image):
        inputs = self.processor(text=["a histopathological image"],images=[image], return_tensors="pt")
        outputs = self.model(**inputs)
        # emb = outputs["vision_model_output"].pooler_output.detach().numpy()
        # emb = emb / np.linalg.norm(emb,axis=1).reshape(-1,1)
        emb = outputs.image_embeds.cpu().detach().numpy()
        return emb

class PLIPTextEmbedding:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("downloaded_data/plip")
        self.processor = CLIPProcessor.from_pretrained("downloaded_data/plip")
    def __call__(self, text):
        inputs = self.processor(text=text,images=[Image.new("RGB", (100,100), color=(0,0,0))], padding = True, return_tensors="pt")
        outputs = self.model(**inputs)
        emb = outputs.text_embeds.cpu().detach().numpy()
        return emb

class CLIPEmbedding:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def __call__(self, image):
        inputs = self.processor(text=["a photo"],images=[image], return_tensors="pt")
        outputs = self.model(**inputs)
        # emb = outputs["vision_model_output"].pooler_output[0].detach().numpy()
        # emb = emb / np.linalg.norm(emb)
        emb = outputs.image_embeds.cpu().detach().numpy()
        
        return emb
    
class InceptionV3Embedding:
    def __init__(self):
        from guided_diffusion.script_util import inceptionv3
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.inception_model = inceptionv3().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __call__(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with th.no_grad():
            self.inception_model.eval()
            emb = self.inception_model(image).cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1).reshape(-1,1)
        return emb
    

if __name__=="__main__":
    if False:
        np.random.seed(0)
        database_dir="mutation_images/frozen_translation_search/database_images"
        query_image_dir="mutation_images/frozen_translation_search/diagnostic_slide_query_images"
        searcher = ImageSearcher(InceptionV3Embedding())
        retrieved_indices, metrics, (precision_per_item_df, classification_prediction_df, cd_per_item_df) = searcher.search(
            database_dir, query_image_dir, use_cache=True
        )
        classification_prediction_df["correct"] = classification_prediction_df["prediction"]==classification_prediction_df["gt"]
        precision_per_item_df["image_class"] = classification_prediction_df["gt"]
        precision_per_class = precision_per_item_df.groupby("image_class")["precision"].mean()
        classification_accuracy_per_class = classification_prediction_df.groupby("gt")["correct"].mean()
        
        print("\n".join(classification_accuracy_per_class.values.astype(str)))
        print("classification_accuracy_per_class.mean()",classification_accuracy_per_class.mean().item())

        print("\n".join(precision_per_class.values.astype(str)))
        print("precision_per_class.mean()", precision_per_class.mean().item())

        print(metrics)
        import pdb;pdb.set_trace()
        
    if False:
        np.random.seed(0)
        query_image_dir="mutation_images/frozen_translation_search/translated_slide_query_images"
        classifier = CLIPClassifier(PLIPTextEmbedding(), PLIPEmbedding())
        retrieved_indices, metrics, classification_prediction_df = classifier.classify(query_image_dir)
        classification_prediction_df["correct"] = classification_prediction_df["prediction"] == classification_prediction_df["gt"]
        
        classification_accuracy_per_class = classification_prediction_df.groupby("gt")["correct"].mean()

        print("\n".join(classification_accuracy_per_class.index.astype(str)))
        print("\n".join(classification_accuracy_per_class.values.astype(str)))
        print("classification_accuracy_per_class.mean()",classification_accuracy_per_class.mean().item())

        print(metrics)
        import pdb;pdb.set_trace()
    
    if False:
        def get_database_dirname(project_id,gene):
            return "mutation_images/TCGA_mutation_images/{}/{}/patch_images_sorted/128x128".format(project_id,gene)

        genes_to_mutate=[
            ("TCGA-COAD",["APC","FBXW7","KRAS"]),
            ("TCGA-ESCA",["TP53","CDKN2A"]),
            ("TCGA-READ",["FBXW7"]),
            ("TCGA-STAD",["TP53"]),
            ("TCGA-OV",["NF1","TP53"]),
            ("TCGA-UCEC",["CTNNB1","PTEN","FBXW7"]),
            ("TCGA-KIRC",["PTEN"]),
            ("TCGA-LUAD",["KRAS","TP53"]),
            ("TCGA-LUSC",["CDKN2A","PIK3CA","TP53"]),
            ("TCGA-BLCA",["FBXW7","TP53","RB1"]),
            ("TCGA-HNSC",["TP53"]),
            ("TCGA-LIHC",["RB1","TP53"]),
            ("TCGA-THCA",["BRAF"]),
        ]

        all_dfs = list()
        for project_id, genes in genes_to_mutate:
            for gene in genes:
                database_dir = get_database_dirname(project_id,gene)
                query_image_dir = "../checkpoints/omics_TCGA_newembed-single_mod_edu_400-128x128-ddib-cmd/translation"

                query_image_fp_df = parse_directory_filenames(query_image_dir)

                query_image_fp_df_nomut = query_image_fp_df.query("(slide_class=='{}')&(translated_slide_class=='{}_0')".format(project_id,gene))
                query_image_fp_df_mut = query_image_fp_df.query("(slide_class=='{}')&(translated_slide_class=='{}_1')".format(project_id,gene))

                searcher = ImageSearcher(InceptionV3Embedding())

                query_image_fn_list = query_image_fp_df_mut["fn"].values.tolist()
                query_image_fp_list = [os.path.join(query_image_dir, fn) for fn in query_image_fn_list]
                query_image_metadata = pd.DataFrame({
                        "image_class": ["mut"]*len(query_image_fp_list)
                    }, 
                    index=query_image_fp_list
                )

                retrieved_indices, metrics, (precision_per_item_df, classification_prediction_df, cd_per_item_df_mut) = searcher.search(
                    database_dir, query_image_dir, 
                    query_image_metadata=query_image_metadata, use_cache=True
                )

                query_image_fn_list = query_image_fp_df_nomut["fn"].values.tolist()
                query_image_fp_list = [os.path.join(query_image_dir, fn) for fn in query_image_fn_list]
                query_image_metadata = pd.DataFrame({
                        "image_class": ["mut"]*len(query_image_fp_list)
                    }, 
                    index=query_image_fp_list
                )

                retrieved_indices, metrics, (precision_per_item_df, classification_prediction_df, cd_per_item_df_nomut) = searcher.search(
                    database_dir, query_image_dir, 
                    query_image_metadata=query_image_metadata, use_cache=True
                )

                d_cosd = (cd_per_item_df_mut["cosd"].mean() - cd_per_item_df_nomut["cosd"].mean()).item()
                d_cosd_pct = d_cosd / (cd_per_item_df_nomut["cosd"].mean() + 1e-7)
                from scipy.stats import wilcoxon
                p = wilcoxon(cd_per_item_df_mut["cosd"].values, cd_per_item_df_nomut["cosd"].values).pvalue

                print("project_id:", project_id, "gene:", gene, "d_cosd:", d_cosd, "d_cosd_pct:", d_cosd_pct, "p: %.2e"%(p))

                cd_per_item_df_stack = cd_per_item_df_nomut.copy()
                cd_per_item_df_stack["project_id"] = project_id
                cd_per_item_df_stack["gene"] = gene
                cd_per_item_df_stack["cosd_mut"] = cd_per_item_df_mut["cosd"].values

                all_dfs.append(cd_per_item_df_stack)
                
        all_dfs = pd.concat(all_dfs)
        
        # save to pickle
        pickle_fp = "mutation_images/frozen_translation_search/cosd_df.pkl"
        with open(pickle_fp, "wb") as f:
            pkl.dump(all_dfs, f)
    
    if True:
        def get_database_dirname(project_id,pathway):
            return "mutation_images/TCGA_pathwayexp_images/{}/{}/patch_images_sorted/128x128".format(project_id,pathway)
        
        pathways_to_manipulate=[
            ("TCGA-COAD",["HALLMARK_MYC_TARGETS_V1","HALLMARK_MYC_TARGETS_V2","HALLMARK_G2M_CHECKPOINT","HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-ESCA",["HALLMARK_MITOTIC_SPINDLE","HALLMARK_MYC_TARGETS_V1"]),
            ("TCGA-READ",["HALLMARK_MYC_TARGETS_V1","HALLMARK_MYC_TARGETS_V2","HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-STAD",["HALLMARK_MYC_TARGETS_V1","HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-OV",  ["HALLMARK_MTORC1_SIGNALING","HALLMARK_GLYCOLYSIS","HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-UCEC",["HALLMARK_MTORC1_SIGNALING","HALLMARK_GLYCOLYSIS","HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-KIRC",["HALLMARK_MTORC1_SIGNALING","HALLMARK_GLYCOLYSIS","HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-LUAD",["HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-LUSC",["HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-BLCA",["HALLMARK_MITOTIC_SPINDLE","HALLMARK_MYC_TARGETS_V1"]),
            ("TCGA-HNSC",["HALLMARK_MITOTIC_SPINDLE"]),
            ("TCGA-LIHC",["HALLMARK_MITOTIC_SPINDLE","HALLMARK_G2M_CHECKPOINT","HALLMARK_MYC_TARGETS_V1"]),
            ("TCGA-THCA",["HALLMARK_MITOTIC_SPINDLE"])
        ]

        all_dfs = list()
        for project_id, pathways in pathways_to_manipulate:
            for pathway in pathways:
                database_dir = get_database_dirname(project_id,pathway)
                query_image_dir = "../checkpoints/omics_TCGA_newembed-trans_manipulations_edu_400-128x128-ddib-cmd/translation"
                query_image_fp_df = parse_directory_filenames(query_image_dir)

                query_image_fp_df_up = query_image_fp_df.query("(slide_class=='{}')&(translated_slide_class=='{}_1')".format(project_id,pathway))
                query_image_fp_df_down = query_image_fp_df.query("(slide_class=='{}')&(translated_slide_class=='{}_0')".format(project_id,pathway))

                searcher = ImageSearcher(InceptionV3Embedding())

                query_image_fn_list = query_image_fp_df_up["fn"].values.tolist()
                query_image_fp_list = [os.path.join(query_image_dir, fn) for fn in query_image_fn_list]

                query_image_metadata = pd.DataFrame({
                        "image_class": ["up"]*len(query_image_fp_list)
                    },
                    index=query_image_fp_list
                )

                retrieved_indices, metrics, (precision_per_item_df, classification_prediction_df, cd_per_item_df_up) = searcher.search(
                    database_dir, query_image_dir, 
                    query_image_metadata=query_image_metadata, use_cache=True
                )

                query_image_fn_list = query_image_fp_df_down["fn"].values.tolist()
                query_image_fp_list = [os.path.join(query_image_dir, fn) for fn in query_image_fn_list]
                query_image_metadata = pd.DataFrame({
                        "image_class": ["up"]*len(query_image_fp_list)
                    }, 
                    index=query_image_fp_list
                )

                retrieved_indices, metrics, (precision_per_item_df, classification_prediction_df, cd_per_item_df_down) = searcher.search(
                    database_dir, query_image_dir, 
                    query_image_metadata=query_image_metadata, use_cache=True
                )

                d_cosd = (cd_per_item_df_up["cosd"].mean() - cd_per_item_df_down["cosd"].mean()).item()
                d_cosd_pct = d_cosd / (cd_per_item_df_down["cosd"].mean() + 1e-7)
                from scipy.stats import wilcoxon
                p = wilcoxon(cd_per_item_df_up["cosd"].values, cd_per_item_df_down["cosd"].values).pvalue

                print("project_id:", project_id, "pathway:", pathway, "d_cosd:", d_cosd, "d_cosd_pct:", d_cosd_pct, "p: %.2e"%(p))

                cd_per_item_df_stack = cd_per_item_df_down.copy()
                cd_per_item_df_stack["project_id"] = project_id
                cd_per_item_df_stack["pathway"] = pathway
                cd_per_item_df_stack["cosd_up"] = cd_per_item_df_up["cosd"].values

                all_dfs.append(cd_per_item_df_stack)
        
        all_dfs = pd.concat(all_dfs)
        # save to pickle
        pickle_fp = "mutation_images/frozen_translation_search/cosd_df_pathway.pkl"
        with open(pickle_fp, "wb") as f:
            pkl.dump(all_dfs, f)




        