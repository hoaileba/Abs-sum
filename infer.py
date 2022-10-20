from T5_copy import T5LongForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from dataloader import CustomDataset
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,)
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers.modeling_outputs import (BaseModelOutput,
                                            BaseModelOutputWithPastAndCrossAttentions,
                                            CausalLMOutputWithCrossAttentions,
                                            Seq2SeqLMOutput,
                                            Seq2SeqModelOutput,)
# data = torch.load("test_long.pt")
# items = torch.stack(data[0])
# items = items.squeeze(1).long().to("cuda")
# print(items.shape)
tokenizer = T5Tokenizer.from_pretrained('t5-small')

model =  T5LongForConditionalGeneration.from_pretrained("t5-small").to("cuda")
checkpoint = torch.load("/home/os_callbot/workspace/hoailb/Text_Sumarization/pegasus/DoAn/exps/T5_decoder_encoder_24062022.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# model1 =  T5LongForConditionalGeneration.from_pretrained("t5-small").to("cuda")
# checkpoint = torch.load("/home/os_callbot/workspace/hoailb/Text_Sumarization/pegasus/DoAn/exps/T5_decoder.pt")
# model1.load_state_dict(checkpoint["model_state_dict"])
checkpoint = None
f = open("train2.txt","w")
with torch.no_grad():
    d = CustomDataset("/home/os_callbot/workspace/hoailb/Text_Sumarization/pegasus/DoAn/data/validation_data/")
    dataloader = DataLoader(
        d,
        batch_size=1,
        num_workers=4,
        shuffle=True
        
    )
    
    # print("items: ",items).
    # max_items = items.argmax(dim = 1)
    # print("max_items: ",max_items)
    # items = items.unsqueeze(0)
    encodings = tokenizer("</s> studies have shown that owning a dog is good for you", truncation=True, padding="max_length",max_length = 100,return_tensors="pt").input_ids
    # print(encodings)
    count = 0
    # print(tokenizer.decode(encodings[0], skip_special_tokens=True))
    # encodings = tokenizer("summarize: studies have shown that owning a dog is good for you", truncation=True, padding="max_length",max_length = 100,return_tensors="pt").input_ids.unsqueeze(0)
    for batch in dataloader:
        count+=1 
        # print(batch[0].shape)
        # print(batch[2][0])
        # ids = batch[0][0]
        # batch[2][0][0] = 18190
        # batch[2][0][1] = 0
        # batch[2][0][2] = 3
        # batch[2][0][3] = 3
        # print(batch[2][0])
        if count == 10:
            break
        # a = model(input_ids = batch[0].to("cuda"),attention_mask = batch[1].to("cuda") ,labels = batch[3].to("cuda"))
        # print(a.logits.argmax(dim=2))
        # print(tokenizer.decode(a.logits.argmax(dim=2)[0], skip_special_tokens=True))
        generated_ids = model.generate(
                input_ids = batch[0].to("cuda"),
                attention_mask = batch[1].to("cuda"),
                num_beams=3,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    max_length = 100
                    )
        # generated_ids_1 = model1.generate(
        #         input_ids = batch[0].to("cuda"),
        #         attention_mask = batch[1].to("cuda"),
        #         num_beams=3,
        #             repetition_penalty=2.5, 
        #             length_penalty=1.0, 
        #             max_length = 100
        #             )
        text = ""
        for line in batch[0][0]:
            if tokenizer.decode(line, skip_special_tokens=True).strip():
                text += tokenizer.decode(line, skip_special_tokens=True).strip()+". "
            # print(tokenizer.decode(line, skip_special_tokens=True))
        print(" --------------------- ")
        print("TEXT: ",text)
        f.write("TEXT: "+text+"\n")
        target = tokenizer.decode(batch[2][0], skip_special_tokens=True)
        print("TARGET: ", target)
        # break
        # print(generated_ids[0])
        f.write("TARGET: "+target+"\n")
        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("PRED DECODER: ",pred)
        f.write("PRED DECODER: "+pred+"\n")
        scores = scorer.score(target, pred)

        print("DECODER SCORE:")
        f.write("DECODER SCORE: \n")
        for k in scores:
            print(k,":")
            f.write(f"{k}:\n")
            print(f"\tprecision: {round(scores[k][0],4)}")
            f.write(f"\tprecision: {round(scores[k][0],4)}\n")
            print(f"\trecall: {round(scores[k][1],4)}")
            f.write(f"\trecall: {round(scores[k][1],4)}\n")
            print(f"\tfmeasure: {round(scores[k][2],4)}")
            f.write(f"\tfmeasure: {round(scores[k][2],4)}\n")
        f.write("------------------------------------------------------------------------------------\n")
        # pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print("PRED DECODER+ENCODER: ",pred)
        
        # scores = scorer.score(target, pred)
        # print("DECODER+ENCODER SCORE:")
        # for k in scores:
        #     print(k,":")
        #     print(f"\tprecision: {round(scores[k][0],4)}")
        #     print(f"\trecall: {round(scores[k][1],4)}")
        #     print(f"\tfmeasure: {round(scores[k][2],4)}")
