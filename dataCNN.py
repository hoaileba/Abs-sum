import datasets,re
import transformers
from transformers import T5Tokenizer
from tqdm import tqdm
tokenizer = T5Tokenizer.from_pretrained('t5-small')
import torch

train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
val_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")
test = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")

# print(train_data)

def process_data_to_model_inputs(path,dataset,max_sample,metadata_path):
    max_len = []
    max_sent = []
    max_token = []
    count = 0
    all_text_embedding = []
    all_text_embedding_decode = []
    all_mask = []
    all_label = []
    f = open(metadata_path,"w")
    for batch in tqdm(dataset):
        if count == max_sample:
            break
        
        article = batch["article"]
        article =  re.sub(r'\.+', ".", article)
        article = article.replace(". .",".")
        article =  re.sub(r"[-}{?/,%@&#$*()=!^><\[\]]", " ", article)
        article = " ".join(article.split())
        highlights = batch["highlights"]
        highlights =  re.sub(r'\.+', ".", highlights)
        highlights = highlights.replace(". .",".")
        highlights =  re.sub(r"[-}{?/,%@&#$*()=!^><\[\]]", " ", highlights)
        highlights = " ".join(highlights.split())
        # print(article
        # print(len(article[0].split(".")))
        max_ = 0
        
        for text in article.split("."):
            
            # if len(text.split(" ")) > 200:
            #     print("-- ",text)
            max_= max(max_,len(text.split(" ")))
        
        if len(article.split(" "))  > 1100 and len(article.split(".")) <= 100 and max_ < 100:

            f.write(f"TEXT: {article}\n")
            f.write(f"TARGET: {highlights}\n--------------------------------------------\n")
            # f.write
            mask = []
            max_token.append(len(highlights.split(" ")))
            max_sent.append(len(highlights.split(".")))
            count+=1
            input_ids = []
            decode_ids = [] 
            # lm_labels = []
            # article = "summarize: ."+article
            for id,text in enumerate(article.split(".")):
                # if id == 0:
                    # text = "summarize: "+text
                encodings = tokenizer(text, truncation=True, padding="max_length",max_length = 100,return_tensors="pt")
                input_ids.append(encodings.input_ids)
                # lm_labels = encodings.inpput_ids
                # lm_labels[encodings.input_ids[:, 1:] == tokenizer.pad_token_id] = -100
                mask.append(1)
            while len(input_ids) < 100:
                add = torch.zeros((1,100))
                mask.append(0)
                input_ids.append(add)
            all_mask.append(mask)
            all_text_embedding.append(input_ids)

            encodings = tokenizer(highlights, truncation=True, padding="max_length",max_length = 100,return_tensors="pt")
            lm_labels = encodings.input_ids
            # print(lm_labels.shape)
            x = lm_labels[:, 0:].clone().detach()
            x[lm_labels[:, 0:] == tokenizer.pad_token_id] = -100

            all_text_embedding_decode.append(encodings.input_ids)
            all_label.append(x)
            for text in highlights.split("."):
                max_len.append(len(text.split(" ")))
            
        # break

            # print(text, len(text.split(" ")))
        # highlight = batch["highlights"]
    print(len(all_text_embedding))
    torch.save(all_text_embedding, f'{path}/X.pt')
    torch.save(all_mask, f'{path}/X_mask.pt')
    torch.save(all_text_embedding_decode, f'{path}/y_ids.pt')
    print(len(all_text_embedding_decode))
    torch.save(all_label, f'{path}/lm_labels.pt')
    # max_len.sort()
    f.close()


if __name__ == "__main__":
    process_data_to_model_inputs("/home/os_callbot/workspace/hoailb/Text_Sumarization/pegasus/DoAn/data/raw",train_data,14000,"trainset1.txt")
    process_data_to_model_inputs("/home/os_callbot/workspace/hoailb/Text_Sumarization/pegasus/DoAn/data/val1",val_data,200,"valset1.txt")