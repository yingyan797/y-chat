from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import torch.nn.functional as Func

def mean_pooling(model_out: torch.Tensor, attn_mask:torch.Tensor):
    token_embed = model_out[0] #First element of model_output contains all token embeddings
    expand_mask = attn_mask.unsqueeze(-1).expand(token_embed.size()).float()
    return torch.sum(token_embed * expand_mask, 1) / torch.clamp(expand_mask.sum(1), min=1e-9)

def sentence_embedding(sentences: list[str]):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_out = model(**encoding)
    embeddings = mean_pooling(model_out, encoding['attention_mask'])
    return Func.normalize(embeddings, p=2, dim=1)

def sentence_similarity(ref: str, candidates: list[str]):
    '''return the similarity score between the input "ref" and each sentence of candidates resectively
    '''
    ref_embed = sentence_embedding([ref])[0]
    cand_embed = sentence_embedding(candidates)
    cos = torch.nn.CosineSimilarity()
    return cos(ref_embed, cand_embed)

def question_entailment(question:str, context:str):
    '''Check whether a question can be answered given some context
    '''
    entail = pipeline("text-classification", model = "cross-encoder/qnli-electra-base", device="cuda")
    return entail(question+","+context)

def answering(question, context):
    qa_model = pipeline("question-answering", device="cuda")
    return qa_model(question = question, context = context)


if __name__ == "__main__":
    # print(sentence_similarity("start having lunch", ["begin to have lunch", "finish cooking meals"]))
    # example output: [0.937, 0.434]
    print(question_entailment("Which capital city is the coldest around the world?", 
        "Climate statistics show that Ulaanbaatar, the capital of Mongolia, has the lowest annual average temperature among all capital cities"))
    # example output: [{'label': 'LABEL_0', 'score': 0.9766839742660522}]

