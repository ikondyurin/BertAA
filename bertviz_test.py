from bertviz import head_view, model_view
import transformers
import torch


def show_head_view(segm00):

    model_version = 'bert-base-cased'
    model = transformers.AutoModelForSequenceClassification.from_pretrained(r"..\BertAA_content\Model\Checkpoints\results_45000\checkpoint-225000", output_attentions=True).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(r"..\BertAA_content\Model\Checkpoints\results_45000\checkpoint-225000")
    text1, text2 = segm00.split("[SEP]")
    inputs = tokenizer(text1, text2, truncation=True, padding='max_length', max_length=255, return_tensors='pt')

    #sentence_a = "The cat sat on the mat"
    #sentence_b = "The cat lay on the rug"
    #inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')

    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    with torch.no_grad():   
        attention = model(**inputs)[-1]
    sentence_b_start = token_type_ids[0].tolist().index(1)
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
    head_view(attention, tokens, sentence_b_start)


if __name__ == "__main__":
    segm00 = """She convinced the guard to let her through the gates and she ran up the familiar stairs until she got to the potted bush next to the front door, 
    she reached down and pulled out the spare key Joey kept there for her. She used them to unlock the door and then tossed her bag down, leaving the door open, and ran upstairs into his room. 
    She then fell down on his bed, held onto a pillow and cried so hard that she didn"t even hear anyone enter the house. "MARY!" Joey yelled, standing dumb-struck in the doorway.[SEP]"Joh..n." 
    her voice turned breathy, heat suffusing through her pores. A loud crash sounded behind them, echoing through the hall. They jerked apart and saw Marcos had accidentally dropped a cement block and created a hole in the floor while Lorna stood glaring at him. 
    Then she bit out,"Nice going laser. As usual I"ll have to fix your mess." Then proceeded to maneuver a few metals plates through the hole in an attempt to mend it."""
    show_head_view(segm00)