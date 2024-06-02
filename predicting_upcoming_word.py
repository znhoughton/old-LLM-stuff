from pprint import pprint
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


### Load model here

#I left this part blank but I can help you with loading the model if you want as well


########################################
model = model.to(device)


#this function takes in a text and provides the probability distribution for the upcoming word (i.e., the probability of the next word for each word)
def to_tokens_and_logprobs(model, tokenizer, input_texts): #Courtesy of this thread: <https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17>
  
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").to(device).input_ids
  
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    
    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch
	
	
#Here's an example usage:

input_texts_alpha = np.array_split(input_texts_alpha, n_batches)
input_texts_alpha = [x.tolist() for x in [*input_texts_alpha]]

batch_alpha = [[]]
timer = 0
for minibatch in input_texts_alpha:
  timer += 1
  print(timer)
  batch_placeholder = to_tokens_and_logprobs(model, tokenizer, minibatch)
  batch_alpha.extend(batch_placeholder)
  

batch_alpha = batch_alpha[1:]
sentence_probs_alpha = [sum(item[1] for item in inner_list[2:]) for inner_list in batch_alpha]