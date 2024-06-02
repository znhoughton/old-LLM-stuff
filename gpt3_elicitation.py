import pandas as pd
from openai import OpenAI
client = OpenAI()
import numpy as np
import random
#import openai

gpt_assistant_prompt = "You are a helpful chat assistant."

temperature=0.2
max_tokens=256
frequency_penalty=0.0

all_sims = []
df = pd.DataFrame()

list_of_binoms = r.binomials_min_and_max #if not using Rmarkdown, convert to pandas object 

for i in range(0,100): #100 sims cuz I'm not rich
  if i % 20 == 0: #troubleshooting
    print(i)
  binoms_response = [] #empty list to save gpt's response
  #for i in 30:
  for i in range(0, len(list_of_binoms['WordA'])): #iterate over each binomial
    #counterbalancing the order so that chatgpt doesn't always get the same ordering.
    counterbalance = random.randint(0, 1) #the counterbalancing works by generating either 0 or 1 randomly, then producing alphabetical order first if 0 is generated, and nonalphabetical order is ordered first if 1 is chosen
    if counterbalance == 0:
      order1 = list_of_binoms['WordA'][i] + ' and ' + list_of_binoms['WordB'][i]
      order2 = list_of_binoms['WordB'][i] + ' and ' + list_of_binoms['WordA'][i]
    else:
      order1 = list_of_binoms['WordB'][i] + ' and ' + list_of_binoms['WordA'][i]
      order2 = list_of_binoms['WordA'][i] + ' and ' + list_of_binoms['WordB'][i]
    
    instructions = "I'm gonna give you two orderings of a pair of words, and I want you to choose whichever one is more natural to you.  Only include the ordering and no other words:" #you'll want to replace this with whatever instructions you want 
    
    instructions_plus_binom = instructions + '\n ' + '"' + order1 + '"' + ' or ' + '"' + order2 + '"' #replace instructions with the target binomial
     
    gpt_prompt = gpt_assistant_prompt, instructions_plus_binom #putting everything together
    
    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": instructions_plus_binom}] #putting everything together 
    
    response = client.chat.completions.create(
    model="gpt-4",			#can change to gpt-3-turbo if you want
    messages = message,
    temperature=temperature, #I kept temperature as 0.2 but I didn't play much around with it, not sure how much of an effect it'll have on results 
    max_tokens=max_tokens,
      frequency_penalty=frequency_penalty #kept this at 0
    )
      
    text = [response.choices[0].message.content] #this is gpt's response 
    #print(text)
    
    binoms_response += text #saving the response in a list
      
  all_sims.append(binoms_response) #list of list, where the inner list is simulations for a given binomial

#all_sims
all_sims_modified = [[word.strip('"') for word in inner_list] for inner_list in all_sims] #remove any quotes around the words

all_sims_modified = [[word.lower() for word in inner_list] for inner_list in all_sims_modified] #same as above

#create the dataframe:
column_names = [f'Binomial_{i+1}' for i in range(len(all_sims_modified[0]))] #cleaning up the column names
