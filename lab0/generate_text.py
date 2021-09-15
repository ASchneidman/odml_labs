from transformers import pipeline
text_generation = pipeline("summarization")
prefix_text = "An apple a day keeps a doctor awway"
generated_text= text_generation(prefix_text,max_length=50, do_sample=False)[0]
print(generated_text)
