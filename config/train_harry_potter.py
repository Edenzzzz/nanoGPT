# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-harrypotter-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'harrypotter-char'
wandb_run_name = 'mini-gpt'

dataset = 'harry_potter'
data_percent = 1.0
gradient_accumulation_steps = 1
batch_size = 64
block_size = 350 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 7000
lr_decay_iters = 7000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
ppl_text = """Chapter 17: The Forbidden Chamber

Harry, Ron, and Hermione stood at the entrance of the dark and foreboding corridor leading to the Forbidden Chamber. The air was thick with tension as they gazed at the ominous door, etched with ancient symbols that seemed to writhe like serpents.

\"We have to find out what's in there, Harry,\" Hermione said, her voice trembling slightly.

Harry nodded, his scar tingling with a sense of foreboding. He knew that whatever lay behind that door was tied to Voldemort's dark past, and they needed to uncover the truth to defeat him once and for all.

With a deep breath, Harry pushed the door open, revealing a chamber bathed in eerie green light. The walls were lined with shelves, each holding a glass vial containing a writhing, shadowy substance.

\"What is this place?\" Ron whispered, his face pale.

\"It's a memory chamber,\" Hermione said, her eyes scanning the vials. \"Voldemort must have stored his most painful memories here, the ones he wanted to forget.\"

As they moved deeper into the chamber, they came across a large, ornate mirror. Harry approached it cautiously, and as he gazed into the glass, he saw a young Tom Riddle, his eyes filled with anger and resentment.

\"This is a Pensieve mirror,\" Hermione explained. \"It allows you to relive memories.\"

Harry reached out and touched the mirror's surface. In an instant, he was transported into a memory from Tom Riddle's past. He watched as Riddle, then a student at Hogwarts, confronted a young Dumbledore about a mysterious artifact he had discovered.

\"The Horcrux,\" Harry muttered, realizing the significance of the memory.

They continued to explore the chamber, uncovering more memories and pieces of Voldemort's dark history. With each revelation, they grew more determined to destroy the Horcruxes and defeat the dark wizard once and for all.

But as they delved deeper into the memories, they also realized the immense power that Voldemort had once possessed and the lengths he would go to achieve his goals. The battle against the dark wizard was far from over, and they would need all the courage and strength they could muster to face the challenges that lay ahead."""
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
