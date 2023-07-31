import logging
import torch
from torch.utils.data import DataLoader

from scipy.special import softmax
from sentence_transformers import SentenceTransformer, models, InputExample, SentencesDataset
from sentence_transformers.util import batch_to_device


class AgreementClassifier:
    def __init__(self, model_path, classifier_path, device='cuda'):
        self.device = device
        self.classifier = torch.load(classifier_path, map_location=device)
        
        word_embedding_model = models.Transformer(model_path, max_seq_length=170)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        
        self.predictor = Predictor(self.model, self.classifier, device=device, show_progress_bar=False)
        
        self.classes = ['disagreement', 'agreement', 'other']
        
    def predict(self, text_pairs, dev_batch_size=16):
        dev_samples = []
        for pair in text_pairs:
            dev_samples.append(InputExample(texts=[pair[0], pair[1]], label=0))
            
        dev_dataset = SentencesDataset(dev_samples, model=self.model)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=dev_batch_size)
        
        preds = self.predictor(dev_dataloader, return_probs=True)
        return preds
        

class Predictor:
    def __init__(self, model, classifier, show_progress_bar, device='cuda:0'):
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() ==\
                                 logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.device = device
        self.classifier = classifier.to(self.device)
        self.model = model.to(self.device)
        self.model.eval()
    
    def __call__(self, dataloader, return_probs=False):
        labels = []

        dataloader.collate_fn = self.model.smart_batching_collate

        iterator = dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert Evaluating")

        for step, (features, label_ids) in enumerate(iterator):
            features = [batch_to_device(sent_features, self.device) for sent_features in features]
            with torch.no_grad():
                rep_a, rep_b = [self.model(sent_features)['sentence_embedding'] for sent_features in features]
            vectors_concat = []
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)
            vectors_concat.append(torch.abs(rep_a - rep_b))
                
            features = torch.cat(vectors_concat, 1)
            output = self.classifier(features)
            labels.extend(output.detach().to("cpu").numpy())
        
        if return_probs:
            return [softmax(el) for el in labels]
        return [np.argmax(el) for el in labels]