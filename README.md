# Triplets

Code behind the experiments for the blog post "Does GPT-4 have common sense?".

You will need the [data from the THINGS triplets dataset](https://osf.io/f5rn6/) and put it into the folder `data/raw`.

Use `make data` to recreate the analysis (costs about 25$ of OpenAI API credits).

Be careful, this uses the OpenAI parallel processor script, so can generate hundreds of requests simultaneously.