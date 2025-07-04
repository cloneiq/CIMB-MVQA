# import os
# import sys
# import os
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer, RobertaTokenizer, RobertaConfig, RobertaModel
from collections import defaultdict, Counter
import torch.nn as nn
import pickle
import time
import logging
from tqdm import tqdm
import random
import os
from .data_tools import colorful_spectrum_mix
import unicodedata
from typing import List, Dict
import h5py

from models.do_question import MedicalQuestionPatternAndEntityExtractor



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VQADataLoader')

for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if env_var in os.environ:
        os.environ.pop(env_var)
os.environ['NO_PROXY'] = '*'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer



class VQADataset(Dataset):

    def __init__(self, data_dir, data_entries, image_dir, transform=None, max_length=32, tokenizer='roberta',
                 answer_vocab=None, mode='train', image_size=384, alpha=1.0, config=None):
        self.config = config
        self.data_dir = data_dir
        self.image_size = image_size
        self.data_entries = data_entries
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.mode = mode
        self.extractor = MedicalQuestionPatternAndEntityExtractor()
        embeddings = np.load(os.path.join(self.config.get('embeddings_dir', 'data/slake/embeddings_all'), 'embeddings.npz'))
        self.pattern_embeddings = embeddings['pattern_embeddings']
        self.entity_value_embeddings = embeddings['entity_value_embeddings']
        with open(os.path.join(self.config.get('embeddings_dir', 'data/slake/embeddings_all'), 'embedding_index.json'), 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        self.pattern_insex = self.index['patterns']
        self.entity_value_index = self.index['entity_values']

        self.image_cache = {}
        self.ae_image_cache = {}
        self.maml_image_cache = {}

        self.alpha = alpha

        self.pre_transform = transforms.Compose([transforms.Resize((image_size, image_size))])

        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load BERT tokenizer
        if tokenizer == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif tokenizer == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', local_files_only=True, weights_only=False)
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_token>']})
            self.visual_token_id = self.tokenizer.convert_tokens_to_ids('<visual_token>')
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer}")

        # Set answer vocabulary
        self.answer_vocab = answer_vocab


    def _get_tokenized(self, idx, question):
        """Get tokenization result"""
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def _load_image(self, img_path):
        """Load image, return multiple sizes, use cache for efficiency"""
        try:
            # Check cache
            if img_path in self.image_cache:
                image = self.image_cache[img_path]
                ae_image = self.ae_image_cache[img_path]
                maml_image = self.maml_image_cache[img_path]
                return image, ae_image, maml_image

            # Load original image
            original_image = Image.open(img_path).convert('RGB')

            # Main image transform
            if self.transform:
                image = self.transform(original_image)
            else:
                image = transforms.ToTensor()(original_image)

            # Autoencoder image (128x128)
            ae_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            ae_image = ae_transform(original_image)

            # MAML image (84x84)
            maml_transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            maml_image = maml_transform(original_image)

            # Update cache
            self.image_cache[img_path] = image
            self.ae_image_cache[img_path] = ae_image
            self.maml_image_cache[img_path] = maml_image

            return image, ae_image, maml_image
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Create default zero tensor as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
            ae_image = torch.zeros(3, 128, 128)
            maml_image = torch.zeros(3, 84, 84)
            return image, ae_image, maml_image
        

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        """Get a single data entry, efficient processing, support frequency-based soft score"""
        entry = self.data_entries[idx]

        # Get current image path and ID
        img_path = os.path.join(self.image_dir, entry['img_name'])
        qid = entry.get('qid', str(idx))  # Use qid if exists, otherwise use index as ID

        # Load image
        image, ae_image, maml_image = self._load_image(img_path)

        # Load and preprocess sample image
        sample_idx = random.randint(0, len(self.data_entries) - 1)
        sample_img_path = os.path.join(self.image_dir, self.data_entries[sample_idx]['img_name'])
        while img_path == sample_img_path:
            sample_idx = random.randint(0, len(self.data_entries) - 1)
            sample_img_path = os.path.join(self.image_dir, self.data_entries[sample_idx]['img_name'])

        # Load and preprocess sample image
        sample_image = Image.open(sample_img_path).convert('RGB')
        sample_image = self.pre_transform(sample_image)
        sample_image = np.array(sample_image)

        or_image = Image.open(img_path).convert('RGB')
        or_image = self.pre_transform(or_image)
        or_image = np.array(or_image)

        # Apply FFT mix
        img21, img12 = colorful_spectrum_mix(or_image, sample_image, alpha=self.alpha, strategy='basic')

        # Convert to PIL image and apply post processing
        img21_pil = Image.fromarray(img21)
        img12_pil = Image.fromarray(img12)
        pos_image = self.post_transform(img21_pil)
        neg_image = self.post_transform(img12_pil)

        # Process question
        question = entry['question']
        if '?' in question:
            do_question = question.replace('?', ' <visual_token>?')
            do_tokenized = self._get_tokenized(idx, do_question)
            do_input_ids = do_tokenized['input_ids']
            do_attention_mask = do_tokenized['attention_mask']
        else:
            do_question = question + ' <visual_token>'
            do_tokenized = self._get_tokenized(idx, do_question)
            do_input_ids = do_tokenized['input_ids']
            do_attention_mask = do_tokenized['attention_mask']
        tokenized = self._get_tokenized(idx, question)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']


        pattern_entity = self.extractor.extract_pattern(question)
        if pattern_entity['syntax_pattern'] in self.pattern_insex:
            pattern_embedding = self.pattern_embeddings[self.pattern_insex[pattern_entity['syntax_pattern']]]
        else:
            pattern_embedding = torch.zeros(768)
        if pattern_entity['core_entity']['value'] in self.entity_value_index:
            entity_embedding = self.entity_value_embeddings[
                self.entity_value_index[pattern_entity['core_entity']['value']]]
        else:
            entity_embedding = torch.zeros(768)
        pattern_embedding = torch.tensor(pattern_embedding, dtype=torch.float32)
        entity_embedding = torch.tensor(entity_embedding, dtype=torch.float32)

        answer_text = preprocess_answer(entry['answer'])
        target = torch.zeros(self.answer_vocab['vocab_size'])
        answer_idx = self.answer_vocab['answer2idx'].get(answer_text, -1)
        scores = self.answer_vocab['answer2score'].get(answer_text, 1.0)
        target.scatter_(0, torch.tensor([answer_idx]), torch.tensor([scores]))

        # Assume image_id and image path are already obtained from data dict
        image_path = img_path

        # Build mask path - in the same directory as the source image
        mask_path = os.path.join(os.path.dirname(image_path), 'mask.png')

        # If mask.png not found, try to find in imgs directory
        if not os.path.exists(mask_path):
            # Try in the same directory as current image
            alt_mask_path = os.path.join(os.path.dirname(image_path), f"mask.png")
            if os.path.exists(alt_mask_path):
                mask_path = alt_mask_path

        # Load and process mask (if exists)
        mask = None
        if os.path.exists(mask_path):
            try:
                # Load mask image
                mask_img = Image.open(mask_path).convert('L')

                # Check if mask is all black (invalid mask)
                mask_array = np.array(mask_img)
                if mask_array.max() > 0:  # Mask is not all black
                    # Create mask transform (only resize, no color transform)
                    mask_transform = transforms.Compose([
                        lambda img: np.array(img) > 0.5,  # Binarize to bool array
                        lambda x: Image.fromarray(x.astype(np.uint8) * 255),
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        lambda x: (x > 0.5).float()
                    ])
                    mask = mask_transform(mask_img)
                else:
                    # Mask is all black, use None
                    mask = torch.zeros((1, self.image_size, self.image_size))
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                mask = torch.zeros((1, self.image_size, self.image_size))
        else:
            # Mask does not exist, use None
            mask = torch.zeros((1, self.image_size, self.image_size))

        # Pack result (ensure qid is returned)
        result = {
            'image': image,
            'pos_image': pos_image,
            'neg_image': neg_image,
            'ae_image': ae_image,
            'maml_image': maml_image,
            'question': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
            'do_question': {
                'input_ids': do_input_ids,
                'attention_mask': do_attention_mask,
            },
            'question_text': question,
            'target': target,  # Frequency-based soft target vector
            'answer_idx': answer_idx,  # Main answer index
            'answer_text': answer_text,  # Main answer text
            'image_path': img_path,
            'mask': mask,  # Add mask
            'pattern_embedding': pattern_embedding,
            'entity_embedding': entity_embedding,
            'qid': qid  # Add question ID for tracking
        }

        # Add other optional fields
        for key in ['location', 'modality', 'qid', 'answer_type', 'content_type']:
            if key in entry:
                result[key] = entry[key]

        return result


class VQADataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.image_dir = config.get('image_dir', os.path.join(self.data_dir, 'slake/imgs'))
        self.train_json = config.get('train_json', os.path.join(self.data_dir, 'slake/train.json'))
        self.val_json = config.get('val_json', os.path.join(self.data_dir, 'slake/validate.json'))
        self.test_json = config.get('test_json', os.path.join(self.data_dir, 'slake/test.json'))
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 16)
        self.image_size = config.get('image_size', 224)
        self.max_length = config.get('max_length', 32)
        self.tokenizer = config.get('tokenizer', 'roberta-base')
        self.min_answer_freq = config.get('min_answer_freq', 5)
        self.rebuild_vocab = config.get('rebuild_vocab', False)
        self.device = config.get('device', 'cuda')

        # Initialization
        self._init_transforms()
        self._load_data()
        self._build_answer_vocab()
        self._init_datasets()
        self._init_loaders()

        # Clean up temporary variables to save memory
        if not config.get('keep_raw_data', False):
            self.train_data = None
            self.val_data = None
            self.test_data = None

    def _init_transforms(self):
        """Initialize image transforms"""
        # Main image transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_data(self):
        """Load datasets, with error handling and format validation"""

        # General loading function
        def load_json(path, name):
            if not os.path.exists(path):
                logger.warning(f"{name} data file does not exist: {path}")
                return []

            logger.info(f"Loading {name} data: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} {name} data entries")
                return data
            except Exception as e:
                logger.error(f"Failed to load {name} data: {e}")
                return []

        # Load each dataset
        self.train_data = load_json(self.train_json, "train")
        self.val_data = load_json(self.val_json, "val")
        self.test_data = load_json(self.test_json, "test")

        # Dataset format validation
        if self.train_data:
            self._validate_data_format(self.train_data[0], "train")
        if self.val_data:
            self._validate_data_format(self.val_data[0], "val")
        if self.test_data:
            self._validate_data_format(self.test_data[0], "test")

    def _validate_data_format(self, entry, name):
        """Validate data format"""
        required_fields = ['question', 'img_name', 'img_id', 'answer']
        for field in required_fields:
            if field not in entry:
                logger.warning(f"{name} data missing required field: {field}")

        logger.info(f"{name} data format: {list(entry.keys())}")

    def _build_answer_vocab(self):
        """Build answer vocabulary, sort by frequency and compute soft scores"""
        vocab_path = os.path.join(self.data_dir, 'answer_vocab.json')

        # If vocab exists and does not need to be rebuilt
        if os.path.exists(vocab_path) and not self.rebuild_vocab:
            print(f"Loading existing answer vocabulary: {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.answer_vocab = json.load(f)

            # Add compatibility code - ensure answer2score key exists
            if 'answer2score' not in self.answer_vocab:
                print("Answer vocabulary missing frequency score info, adding...")
                self.answer_vocab['answer2freq'] = {}
                self.answer_vocab['answer2score'] = {}

                # Add default frequency score for all answers
                for ans in self.answer_vocab['answer2idx'].keys():
                    if ans != '<UNK>':  # Skip UNK token
                        # Use default frequency 1 (score 0.3)
                        self.answer_vocab['answer2freq'][ans] = 1
                        self.answer_vocab['answer2score'][ans] = 0.3

                # Save updated vocabulary
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.answer_vocab, f, ensure_ascii=False, indent=2)

            print(f"Answer vocabulary size: {self.answer_vocab['vocab_size']}")
            return

        print("Building new answer vocabulary...")

        answer_counter: Counter[str] = Counter()
        norm2raw: Dict[str, str] = {}  # normalized key -> first occurrence of original form
        answer_idx2text: Dict[int, str] = {}

        def process_dataset(dataset: List[Dict], name: str) -> int:
            if not dataset:
                return 0
            cnt = 0
            for item in dataset:
                answer_key = "answer" if "answer" in item else "a" if "a" in item else None
                if not answer_key:
                    continue

                ans_field = item[answer_key]

                norm = preprocess_answer(ans_field)
                answer_counter[norm] += 1
                norm2raw.setdefault(norm, ans_field)
                cnt += 1
            return cnt

        train_cnt = process_dataset(self.train_data, "train")
        val_cnt = process_dataset(self.val_data, "val")
        test_cnt = process_dataset(self.test_data, "test")

        print(f"Collected {train_cnt} answers from training set")
        print(f"Collected {val_cnt} answers from validation set")
        print(f"Collected {test_cnt} answers from test set")
        print(f"Collected {len(answer_counter)} unique normalized answers in total")

        # ---- Sort by frequency ----------------------------------------------------- #
        sorted_answers = sorted(answer_counter.items(), key=lambda kv: kv[1], reverse=True)
        print("Top 10 high-frequency answers (normalized):", sorted_answers[:10])

        answer2idx: Dict[str, int] = {}
        idx2answer: Dict[int, str] = {}
        answer2freq: Dict[str, int] = {}
        answer2score: Dict[str, float] = {}

        # --- Generate mapping & soft scores ---------------------------------------- #
        for i, (norm_ans, freq) in enumerate(sorted_answers):
            answer2idx[norm_ans] = i
            # Keep the first occurrence of the original form for display / reverse mapping
            idx2answer[i] = norm2raw[norm_ans]
            answer2freq[norm_ans] = freq

            # Soft score rules
            if freq == 0:
                score = 0.0
            elif freq == 1:
                score = 0.3
            elif freq == 2:
                score = 0.6
            elif freq == 3:
                score = 0.9
            else:
                score = 1.0
            answer2score[norm_ans] = score

        # ---- Aggregate and save --------------------------------------------------- #
        self.answer_vocab = {
            "answer2idx": answer2idx,
            "idx2answer": idx2answer,
            "answer2freq": answer2freq,
            "answer2score": answer2score,
            "vocab_size": len(answer2idx),
        }

        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Saving answer vocabulary to: {vocab_path}")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.answer_vocab, f, ensure_ascii=False, indent=2)

    def _init_datasets(self):
        """Initialize dataset objects"""
        # Training set
        if self.train_data:
            self.train_dataset = VQADataset(
                self.data_dir,
                self.train_data,
                self.image_dir,
                transform=self.transform,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='train',
                image_size=self.image_size,
                config=self.config
            )
            logger.info(f"Training set size: {len(self.train_dataset)}")
        else:
            logger.warning("Training set is empty")
            self.train_dataset = None

        # Validation set
        if self.val_data:
            self.val_dataset = VQADataset(
                self.data_dir,
                self.val_data,
                self.image_dir,
                transform=self.transform,  # Validation set uses basic transform
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='val',
                image_size=self.image_size,
                config=self.config
            )
            logger.info(f"Validation set size: {len(self.val_dataset)}")
        else:
            logger.warning("Validation set is empty")
            self.val_dataset = None

        # Test set
        if self.test_data:
            self.test_dataset = VQADataset(
                self.data_dir,
                self.test_data,
                self.image_dir,
                transform=self.transform,  # Test set uses basic transform
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='test',
                image_size=self.image_size,
                config=self.config
            )
            logger.info(f"Test set size: {len(self.test_dataset)}")
        else:
            logger.info("Test set not provided")

    def _init_loaders(self):
        """Initialize data loaders"""
        # Create training loader
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                prefetch_factor=2 if self.num_workers > 0 else None,  # Prefetch factor
                persistent_workers=self.num_workers > 0  # Keep worker processes alive
            )

        # Create validation loader - use single-process mode to avoid Windows serialization issues
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,  # Use 0 for single-process mode to avoid Windows multi-process issues
                persistent_workers=False,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

        # Create test loader - also use single-process mode
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,  # Use single-process mode
                persistent_workers=False,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

    def collate_fn(self, batch):
        # Basic batch data collection
        images = torch.stack([item['image'] for item in batch])
        pos_images = torch.stack([item['pos_image'] for item in batch])
        neg_images = torch.stack([item['neg_image'] for item in batch])
        ae_images = torch.stack([item['ae_image'] for item in batch])
        maml_images = torch.stack([item['maml_image'] for item in batch])
        input_ids = torch.stack([item['question']['input_ids'] for item in batch])
        attention_mask = torch.stack([item['question']['attention_mask'] for item in batch])
        do_input_ids = torch.stack([item['do_question']['input_ids'] for item in batch])
        do_attention_mask = torch.stack([item['do_question']['attention_mask'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        mask = torch.stack([item['mask'] for item in batch])
        answer_indices = torch.tensor([item['answer_idx'] for item in batch], dtype=torch.float32)
        question_texts = [item['question_text'] for item in batch]
        answer_texts = [item['answer_text'] for item in batch]
        answer_types = [item.get('answer_type', '') for item in batch]
        image_paths = [item['image_path'] for item in batch]

        pattern_embedding = torch.stack([item['pattern_embedding'] for item in batch])
        entity_embedding = torch.stack([item['entity_embedding'] for item in batch])
        
        # Build the complete batch (excluding image batch)
        main_batch = {
            'images': images,
            'pos_images': pos_images,
            'neg_images': neg_images,
            'ae_images': ae_images,
            'maml_images': maml_images,
            'questions': {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            'do_questions': {
                'input_ids': do_input_ids,
                'attention_mask': do_attention_mask
            },
            'targets': targets,
            'answer_indices': answer_indices,
            'question_texts': question_texts,
            'answer_texts': answer_texts,
            'answer_types': answer_types,
            'image_paths': image_paths,
            'mask': mask,
            'pattern_embedding': pattern_embedding,
            'entity_embedding': entity_embedding
        }

        return main_batch

    def get_loaders(self):
        """Get all data loaders"""
        loaders = {}

        if hasattr(self, 'train_loader'):
            loaders['train'] = self.train_loader

        if hasattr(self, 'val_loader'):
            loaders['val'] = self.val_loader

        if hasattr(self, 'test_loader'):
            loaders['test'] = self.test_loader

        return loaders

    def get_answer_vocab(self):
        """Get answer vocabulary"""
        return self.answer_vocab

    def idx2answer(self, idx):
        """Convert index to answer text"""
        if isinstance(idx, int):
            idx_str = str(idx)
        else:
            idx_str = idx

        return self.answer_vocab['idx2answer'].get(idx_str, '<UNK>')

    def answer2idx(self, answer):
        """Convert answer text to index"""
        return self.answer_vocab['answer2idx'].get(answer, self.answer_vocab['answer2idx']['<UNK>'])

    def _load_json_data(self, json_path):

        logger.info(f"Loading data from {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = []
        for item in data:
            if 'answers' in item and isinstance(item['answers'], list):
                # Handle multiple answers
                answers = [a['answer'] for a in item['answers']]
                scores = [a.get('answer_confidence', 1.0) for a in item['answers']]

                # Compute average score for each unique answer
                answer_scores = {}
                for ans, score in zip(answers, scores):
                    if ans not in answer_scores:
                        answer_scores[ans] = []
                    answer_scores[ans].append(score)

                # Merge into unique answer list and their average scores
                unique_answers = list(answer_scores.keys())
                avg_scores = [sum(answer_scores[ans]) / len(answer_scores[ans]) for ans in unique_answers]

                # Normalize scores
                total = sum(avg_scores)
                if total > 0:
                    avg_scores = [s / total for s in avg_scores]

                # Convert to our format
                labels = []
                for ans in unique_answers:
                    if ans in self.answer_vocab['answer2idx']:
                        labels.append(self.answer_vocab['answer2idx'][ans])

                answer_obj = {
                    'labels': labels,
                    'scores': avg_scores
                }
                item['answer'] = answer_obj

            entries.append(item)

        logger.info(f"Loaded {len(entries)} data entries")
        return entries

