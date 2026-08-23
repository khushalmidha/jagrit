import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

class NRMSDataset(Dataset):
    def __init__(self, behaviors_path, news_path, word2idx, max_title_len=20, max_history_len=30, num_negatives=2, is_train=True):
        self.word2idx = word2idx
        self.max_title_len = max_title_len
        self.max_history_len = max_history_len
        self.num_negatives = num_negatives
        self.is_train = is_train
        
        # Load news titles
        news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        df_news = pd.read_csv(news_path, sep='\t', header=None, names=news_cols)
        
        self.news2title = {}
        # padding news vector
        self.news2title['<PAD>'] = [0] * self.max_title_len
        
        # all available news ids (for padding negatives if needed)
        self.all_news_ids = []
        
        for _, row in df_news.iterrows():
            news_id = row['news_id']
            title = str(row['title'])
            tokens = self._tokenize(title)
            self.news2title[news_id] = tokens
            self.all_news_ids.append(news_id)
            
        # Load behaviors
        behavior_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        df_behaviors = pd.read_csv(behaviors_path, sep='\t', header=None, names=behavior_cols)
        
        self.samples = []
        
        for _, row in df_behaviors.iterrows():
            imp_id = row['impression_id']
            history_str = str(row['history'])
            if history_str == 'nan':
                history = []
            else:
                history = history_str.split()
                
            impressions_str = str(row['impressions'])
            if impressions_str == 'nan':
                continue
            
            impressions = impressions_str.split()
            
            if self.is_train:
                positives = [imp.split('-')[0] for imp in impressions if imp.endswith('-1')]
                negatives = [imp.split('-')[0] for imp in impressions if imp.endswith('-0')]
                
                for pos in positives:
                    # sample K negatives from the SAME impression
                    if len(negatives) >= self.num_negatives:
                        sampled_negatives = random.sample(negatives, self.num_negatives)
                    else:
                        # MIND-standard practice: pad with randomly sampled negatives from elsewhere
                        sampled_negatives = negatives[:]
                        while len(sampled_negatives) < self.num_negatives:
                            random_neg = random.choice(self.all_news_ids)
                            if random_neg != pos and random_neg not in sampled_negatives:
                                sampled_negatives.append(random_neg)
                                
                    candidates = [pos] + sampled_negatives
                    # label is the index of the positive (always 0 here, but we will shuffle)
                    labeled_candidates = list(zip(candidates, [1] + [0]*self.num_negatives))
                    random.shuffle(labeled_candidates)
                    
                    shuffled_candidates = [c for c, l in labeled_candidates]
                    label_idx = [l for c, l in labeled_candidates].index(1)
                    
                    self.samples.append({
                        'history': history,
                        'candidates': shuffled_candidates,
                        'label': label_idx
                    })
            else:
                # For dev/eval, evaluate on the full real impression without negative sampling
                candidates = [imp.split('-')[0] for imp in impressions]
                labels = [int(imp.split('-')[1]) for imp in impressions]
                self.samples.append({
                    'impression_id': imp_id,
                    'history': history,
                    'candidates': candidates,
                    'labels': labels
                })
                
    def _tokenize(self, text):
        words = text.lower().split()
        words = [w.strip('.,!?"\'():;') for w in words]
        words = [w for w in words if w]
        
        tokens = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        if len(tokens) > self.max_title_len:
            tokens = tokens[:self.max_title_len]
        else:
            tokens = tokens + [self.word2idx['<PAD>']] * (self.max_title_len - len(tokens))
        return tokens
        
    def _get_history_tensors(self, history):
        hist_tensors = []
        for h in history[-self.max_history_len:]:
            hist_tensors.append(self.news2title.get(h, self.news2title['<PAD>']))
            
        # pad history
        while len(hist_tensors) < self.max_history_len:
            hist_tensors.insert(0, self.news2title['<PAD>']) # prepend padding for sequence
            
        return torch.tensor(hist_tensors, dtype=torch.long)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        history_tensor = self._get_history_tensors(sample['history'])
        
        candidate_tensors = []
        for c in sample['candidates']:
            candidate_tensors.append(self.news2title.get(c, self.news2title['<PAD>']))
        candidate_tensor = torch.tensor(candidate_tensors, dtype=torch.long)
        
        if self.is_train:
            return history_tensor, candidate_tensor, sample['label']
        else:
            # return lists and simple types so custom collate is not strictly needed for eval if batch_size=1
            return {
                'impression_id': sample['impression_id'],
                'history': history_tensor,
                'candidates': candidate_tensor,
                'labels': torch.tensor(sample['labels'], dtype=torch.float32)
            }
