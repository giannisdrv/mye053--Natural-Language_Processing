import gc
import pandas as pd
import re
import string
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import nltk
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB, MultinomialNB, GaussianNB
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight

nltk.download('stopwords')
nltk.download('snowball_data')

stemmer = nltk.SnowballStemmer("english")
stop_words = set(nltk.corpus.stopwords.words('english'))

custom_stopwords = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 
    'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'fsis', 'usda', 'recall'
}
stop_words = stop_words.union(custom_stopwords)

labels = [
    'case number:', 'date opened:', 'date closed:', 'recall class:', 'press release (y/n):',
    'domestic est. number:', 'imported product (y/n):', 'foreign estab. number:', 'city:',
    'state:', 'country:', 'total pounds recalled:', 'pounds recovered:',
    'for immediate release', 'end of company transmission', 
    'media inquiries:', 'this archive of previously issued food recalls',
    'product details', 'risk statement', 'action taken by the company', 'point of sale notices',
    'date published', 'product description', 'identifying features', 'what are the defects?', 
    'what are the hazards?', 'what should consumers do?', 'for further information contact',
    'reason for recall:', 'hazard classification:', 'company / firm:', 'distribution:', 
    'extent of the distribution:', 'reference number:', 'affected products', 'brand name', 
    'common name', 'size', 'upc', 'code(s) on product', 'editor’s note:', 'washington', 'page content', 
    'this archive of previously issued food recalls and allergy alerts is provided for reference and research purposes.',
    'users should note that the products listed in the archive have been subject to removal from the marketplace'
]

def clean_text(text, isbert):
    text = str(text)
    lines = text.split('\n')
    unique_lines = []
    seen = set()

    for line in lines:
        clean_line = line.strip()
        if clean_line and clean_line not in seen:
            seen.add(clean_line)
            unique_lines.append(clean_line)
    text = ' '.join(unique_lines)

    for label in labels:
        text = text.replace(label, '')

    if not isbert:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text)
        
        words = text.split()
        cleaned_words = []
        for word in words:
            if word not in stop_words and len(word) > 1:          
                stemmed_word = stemmer.stem(word)  
                cleaned_words.append(stemmed_word)
        text = ' '.join(cleaned_words).strip()
        text = text.lower()
        text = re.sub(r'\d+', ' ', text)
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)

    return text

def calculate_competition_score(y_true_haz, y_pred_haz, y_true_prod, y_pred_prod):
    f1_hazard = f1_score(y_true_haz, y_pred_haz, average='macro', zero_division=0)
    correct_hazard_mask = (np.array(y_true_haz) == np.array(y_pred_haz))
    
    y_true_prod_filtered = np.array(y_true_prod)[correct_hazard_mask]
    y_pred_prod_filtered = np.array(y_pred_prod)[correct_hazard_mask]
    
    all_product_classes = np.unique(y_true_prod)
    if len(y_true_prod_filtered) > 0:
        f1_product_conditional = f1_score(
            y_true_prod_filtered, y_pred_prod_filtered, 
            labels=all_product_classes, average='macro', zero_division=0
        )
    else:
        f1_product_conditional = 0.0
        
    final_score = (f1_hazard + f1_product_conditional) / 2.0
    
    print("\n HAZARD CATEGORY - Classification Report:")
    print(classification_report(y_true_haz, y_pred_haz, zero_division=0))
    print("\n HAZARD CATEGORY - Confusion Matrix:")
    print(confusion_matrix(y_true_haz, y_pred_haz))
    
    print("\n PRODUCT CATEGORY (Conditional) - Classification Report:")
    if len(y_true_prod_filtered) > 0:
        print(classification_report(y_true_prod_filtered, y_pred_prod_filtered, zero_division=0))
    else:
        print("Couldn't find hazard category to predict the product category.\n")

    print(f"F1 Hazard Category:             {f1_hazard:.4f}")
    print(f"F1 Product:       {f1_product_conditional:.4f}")
    print(f"Final Score (Score):           {final_score:.4f}\n")



def logistic_regression_model(X_train, y_train_hazard, y_train_product, X_valid_tfidf, y_valid_haz, y_valid_prod):
    lr_hazard_model = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    lr_hazard_model.fit(X_train, y_train_hazard)
    lr_product_model = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    lr_product_model.fit(X_train, y_train_product)
    
    y_pred_haz = lr_hazard_model.predict(X_valid_tfidf)
    y_pred_prod = lr_product_model.predict(X_valid_tfidf)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return lr_hazard_model, lr_product_model

def ComplementBayes(X_train, y_train_hazard, y_train_product, X_valid_tfidf, y_valid_haz, y_valid_prod):
    nb_comp_hazard_model = ComplementNB()
    nb_comp_hazard_model.fit(X_train, y_train_hazard)
    nb_comp_product_model = ComplementNB()
    nb_comp_product_model.fit(X_train, y_train_product)

    y_pred_haz = nb_comp_hazard_model.predict(X_valid_tfidf)
    y_pred_prod = nb_comp_product_model.predict(X_valid_tfidf)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return nb_comp_hazard_model, nb_comp_product_model

def MultinomialBayes(X_train, y_train_hazard, y_train_product, X_valid_tfidf, y_valid_haz, y_valid_prod):
    nb_mult_hazard_model = MultinomialNB()
    nb_mult_hazard_model.fit(X_train, y_train_hazard)
    nb_mult_product_model = MultinomialNB()
    nb_mult_product_model.fit(X_train, y_train_product)

    y_pred_haz = nb_mult_hazard_model.predict(X_valid_tfidf)
    y_pred_prod = nb_mult_product_model.predict(X_valid_tfidf)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return nb_mult_hazard_model, nb_mult_product_model

def SVD_GaussianBayes(X_train, y_train_hazard, y_train_product, X_valid_tfidf, y_valid_haz, y_valid_prod):
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_valid_svd = svd.transform(X_valid_tfidf)

    nb_gaussian_hazard_model = GaussianNB()
    nb_gaussian_hazard_model.fit(X_train_svd, y_train_hazard)
    nb_gaussian_product_model = GaussianNB()
    nb_gaussian_product_model.fit(X_train_svd, y_train_product)

    y_pred_haz = nb_gaussian_hazard_model.predict(X_valid_svd)
    y_pred_prod = nb_gaussian_product_model.predict(X_valid_svd)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return svd, nb_gaussian_hazard_model, nb_gaussian_product_model

def get_doc_vector(tokens, model, vector_size):
    valid_words = [word for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)

def Word2Vec_GaussianBayes(X_train, y_train_hazard, y_train_product, X_valid, y_valid_haz, y_valid_prod):
    sentences_train = [text.split() for text in X_train]
    sentences_valid = [text.split() for text in X_valid]
    w2v_model = Word2Vec(sentences=sentences_train, vector_size=100, window=5, min_count=2, sg=1, workers=4, seed=42)

    X_train_w2v = np.array([get_doc_vector(tokens, w2v_model, 100) for tokens in sentences_train])
    X_valid_w2v = np.array([get_doc_vector(tokens, w2v_model, 100) for tokens in sentences_valid])

    gnb_hazard_model = GaussianNB()
    gnb_hazard_model.fit(X_train_w2v, y_train_hazard)
    gnb_product_model = GaussianNB()
    gnb_product_model.fit(X_train_w2v, y_train_product)

    y_pred_haz = gnb_hazard_model.predict(X_valid_w2v)
    y_pred_prod = gnb_product_model.predict(X_valid_w2v)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return w2v_model, gnb_hazard_model, gnb_product_model

def Tuned_LogisticRegression(X_train, y_train_hazard, y_train_product, X_valid_tfidf, y_valid_haz, y_valid_prod):
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
    base_lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    
    grid_hazard = GridSearchCV(base_lr, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_hazard.fit(X_train, y_train_hazard)
    best_lr_hazard = grid_hazard.best_estimator_

    grid_product = GridSearchCV(base_lr, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_product.fit(X_train, y_train_product)
    best_lr_product = grid_product.best_estimator_

    y_pred_haz = best_lr_hazard.predict(X_valid_tfidf)
    y_pred_prod = best_lr_product.predict(X_valid_tfidf)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return best_lr_hazard, best_lr_product, grid_hazard.best_params_['C'], grid_product.best_params_['C']

def Ensemble_VotingClassifier(X_train, y_train_hazard, y_train_product, X_valid_tfidf, y_valid_haz, y_valid_prod, best_C_haz, best_C_prod):
    lr_haz = LogisticRegression(C=best_C_haz, class_weight='balanced', max_iter=2000, random_state=42)
    nb_haz = ComplementNB()
    ensemble_haz = VotingClassifier(estimators=[('lr', lr_haz), ('nb', nb_haz)], voting='soft')
    ensemble_haz.fit(X_train, y_train_hazard)

    lr_prod = LogisticRegression(C=best_C_prod, class_weight='balanced', max_iter=2000, random_state=42)
    nb_prod = ComplementNB()
    ensemble_prod = VotingClassifier(estimators=[('lr', lr_prod), ('nb', nb_prod)], voting='soft')
    ensemble_prod.fit(X_train, y_train_product)

    y_pred_haz = ensemble_haz.predict(X_valid_tfidf)
    y_pred_prod = ensemble_prod.predict(X_valid_tfidf)
    calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)
    
    return ensemble_haz, ensemble_prod

def create_classical_submission(choice, haz_model, prod_model, feature_extractor):
    print("\n Creating submission.csv \n")
    df_test = pd.read_csv('test.csv')
    
    cleaned_test = [
        clean_text(str(title), isbert=False) + ' ' + clean_text(str(text), isbert=False)
        for title, text in zip(df_test['title'], df_test['text'])
    ]

    if choice in {"1", "2", "3", "6", "7"}:
        tfidf = feature_extractor
        X_test_feats = tfidf.transform(cleaned_test)
        
    elif choice == "4":
        tfidf, svd = feature_extractor
        X_test_feats = tfidf.transform(cleaned_test)
        X_test_feats = svd.transform(X_test_feats)
        
    elif choice == "5":
        w2v_model = feature_extractor
        sentences_test = [text.split() for text in cleaned_test]
        X_test_feats = np.array([get_doc_vector(tokens, w2v_model, 100) for tokens in sentences_test])

    final_pred_haz = haz_model.predict(X_test_feats)
    final_pred_prod = prod_model.predict(X_test_feats)

    submission_df = pd.DataFrame({
        'id': df_test['id'],
        'hazard-category': final_pred_haz,
        'product-category': final_pred_prod
    })
    submission_df.to_csv('submission.csv', index=False)
    print(" Submission.csv saved successfully!")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

class FocalLossTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics, class_weights, gamma):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            compute_metrics=compute_metrics
        )
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss_unweighted = F.cross_entropy(
            logits.view(-1, self.model.config.num_labels), 
            labels.view(-1), 
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss_unweighted)
        pt = torch.clamp(pt, min=1e-5, max=0.99999)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_unweighted
        
        if self.class_weights is not None:
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            batch_weights = weights[labels.view(-1)]
            focal_loss = focal_loss * batch_weights
        
        loss = focal_loss.mean()

        return (loss, outputs) if return_outputs else loss

def train_transformer(X_train, y_train, X_valid, y_valid, task_name="hazard", model_name="roberta-base"):
    le = LabelEncoder()
    all_labels = list(y_train) + list(y_valid)
    le.fit(all_labels)
    y_train_encoded = le.transform(y_train)
    y_valid_encoded = le.transform(y_valid)
    num_classes = len(le.classes_)

    unique_classes = np.unique(y_train_encoded)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_encoded)
    class_weights_tensor = torch.clamp(torch.tensor(weights, dtype=torch.float32), min=1.0, max=30.0)
    
    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train_encoded})
    valid_dataset = Dataset.from_dict({'text': X_valid, 'label': y_valid_encoded})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    training_args = TrainingArguments(
        output_dir=f'./results_{task_name}',    
        num_train_epochs=10,              
        per_device_train_batch_size=32,   
        per_device_eval_batch_size=32,    
        warmup_ratio=0.1,                
        weight_decay=0.01,               
        logging_dir=f'./logs_{task_name}',            
        eval_strategy="epoch",     
        save_strategy="epoch",                         
        load_best_model_at_end=True,                 
        learning_rate=5e-6,
        metric_for_best_model="f1",          
        greater_is_better=True,
        save_total_limit=1,
        bf16 = True,
        logging_steps=10 
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        gamma=2.0  
    )

    trainer.train()
    predictions = trainer.predict(tokenized_valid)
    y_pred_encoded = np.argmax(predictions.predictions, axis=-1)
    
    return le.inverse_transform(y_pred_encoded), trainer, le

def FineTune_BERT(X_train, y_train_hazard, y_train_product, X_valid, y_valid_haz, y_valid_prod):
    print("\n Training for Hazard category \n")
    y_pred_haz, trainer_haz, le_haz = train_transformer(X_train, y_train_hazard, X_valid, y_valid_haz, task_name="hazard")

    print("\n Training for Product category \n")
    y_pred_prod, trainer_prod, le_prod = train_transformer(X_train, y_train_product, X_valid, y_valid_prod, task_name="product")
    
    return y_pred_haz, y_pred_prod, trainer_haz, le_haz, trainer_prod, le_prod

def create_roberta_submission(trainer_haz, le_haz, trainer_prod, le_prod, test_path='test.csv', model_name="roberta-base"):
    print("\n Creating submission.csv \n")
    df_test = pd.read_csv(test_path, engine='python', on_bad_lines='skip') 
    
    X_test = [clean_text(str(title), isbert=True) + ' ' + clean_text(str(text), isbert=True)
              for title, text in zip(df_test['title'].tolist(), df_test['text'].tolist())]
    
    test_dataset = Dataset.from_dict({'text': X_test, 'label': [0] * len(X_test)})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    preds_haz = trainer_haz.predict(tokenized_test)
    final_pred_haz = le_haz.inverse_transform(np.argmax(preds_haz.predictions, axis=-1))

    preds_prod = trainer_prod.predict(tokenized_test)
    final_pred_prod = le_prod.inverse_transform(np.argmax(preds_prod.predictions, axis=-1))
    
    submission_df = pd.DataFrame({
        'id': df_test['id'],
        'hazard-category': final_pred_haz,
        'product-category': final_pred_prod
    })
    submission_df.to_csv('submission.csv', index=False)
    print(" Submission.csv saved successfully.")



MODELS = {
    "1": "Logistic Regression",
    "2": "Complement Naive Bayes",
    "3": "Multinomial Naive Bayes",
    "4": "SVD + Gaussian Naive Bayes",
    "5": "Word2Vec + Gaussian Naive Bayes",
    "6": "Tuned Logistic Regression",
    "7": "Ensemble (Tuned LR + Complement NB)",
    "8": "RoBERTa (Fine-tune)",
}

TFIDF_MODELS  = {"1", "2", "3", "4", "6", "7"}   
BERT_MODELS   = {"8"}                              

def prompt_model_choice():
    print("\n  Food Hazard Detection — Model Selection \n")
    for key, name in MODELS.items():
        print(f"  [{key}] {name}")
    print("\n" )
    while True:
        choice = input("Select a model (1-8): ").strip()
        if choice in MODELS:
            print(f"\n  Running: {MODELS[choice]}\n")
            return choice
        print("  Invalid choice, please enter a number between 1 and 8.")

def build_tfidf(X_train_raw, X_valid_raw):
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(
        min_df=5, max_df=0.85, ngram_range=(1, 2)
    )
    X_train = tfidf.fit_transform(X_train_raw)
    X_valid = tfidf.transform(X_valid_raw)
    return tfidf, X_train, X_valid

def main():
    choice = prompt_model_choice()

    train_df = pd.read_csv('train.csv', engine='python', on_bad_lines='skip')
    valid_df = pd.read_csv('valid.csv', engine='python', on_bad_lines='skip')

    y_train_haz  = train_df['hazard-category'].tolist()
    y_train_prod = train_df['product-category'].tolist()
    y_valid_haz  = valid_df['hazard-category'].tolist()
    y_valid_prod = valid_df['product-category'].tolist()

    if choice in BERT_MODELS:
        print("Cleaning text for RoBERTa \n")
        X_train = [
            clean_text(str(title), isbert=True) + ' ' + clean_text(str(text), isbert=True)
            for title, text in zip(train_df['title'], train_df['text'])
        ]
        X_valid = [
            clean_text(str(title), isbert=True) + ' ' + clean_text(str(text), isbert=True)
            for title, text in zip(valid_df['title'], valid_df['text'])
        ]
    else:
        print("Cleaning text for classical models \n")
        X_train_raw = [
            clean_text(str(title), isbert=False) + ' ' + clean_text(str(text), isbert=False)
            for title, text in zip(train_df['title'], train_df['text'])
        ]
        X_valid_raw = [
            clean_text(str(title), isbert=False) + ' ' + clean_text(str(text), isbert=False)
            for title, text in zip(valid_df['title'], valid_df['text'])
        ]

    print(f"Training on {len(train_df)} examples, validating on {len(valid_df)}.\n")

    if choice in TFIDF_MODELS:
        print("Building TF-IDF matrix \n")
        tfidf, X_train, X_valid = build_tfidf(X_train_raw, X_valid_raw)

    haz_m, prod_m, feature_ext = None, None, None

    if choice == "1":
        haz_m, prod_m = logistic_regression_model(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod)
        feature_ext = tfidf
    elif choice == "2":
        haz_m, prod_m = ComplementBayes(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod)
        feature_ext = tfidf
    elif choice == "3":
        haz_m, prod_m = MultinomialBayes(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod)
        feature_ext = tfidf
    elif choice == "4":
        svd, haz_m, prod_m = SVD_GaussianBayes(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod)
        feature_ext = (tfidf, svd)
    elif choice == "5":
        w2v, haz_m, prod_m = Word2Vec_GaussianBayes(X_train_raw, y_train_haz, y_train_prod, X_valid_raw, y_valid_haz, y_valid_prod)
        feature_ext = w2v
    elif choice == "6":
        haz_m, prod_m, _, _ = Tuned_LogisticRegression(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod)
        feature_ext = tfidf
    elif choice == "7":
        _, _, best_C_haz, best_C_prod = Tuned_LogisticRegression(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod)
        haz_m, prod_m = Ensemble_VotingClassifier(X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod, best_C_haz, best_C_prod)
        feature_ext = tfidf
    elif choice == "8":
        torch.cuda.empty_cache()
        gc.collect()
        y_pred_haz, y_pred_prod, trainer_haz, le_haz, trainer_prod, le_prod = FineTune_BERT(
            X_train, y_train_haz, y_train_prod, X_valid, y_valid_haz, y_valid_prod
        )
        calculate_competition_score(y_valid_haz, y_pred_haz, y_valid_prod, y_pred_prod)

    answer = input(f"\nCreate submission.csv with {MODELS[choice]}? (y/n): ").strip().lower()
    
    if answer == 'y':
        if choice in BERT_MODELS:
            create_roberta_submission(trainer_haz, le_haz, trainer_prod, le_prod)
        else:
            create_classical_submission(choice, haz_m, prod_m, feature_ext)

if __name__ == "__main__":
    main()