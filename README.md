# Read Me

<br/>

**Paper:** Sentence-level Media Bias Analysis Informed by Discourse Structures<br/>
**Accepted:** The 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang, Lu Wang, Nick Beauchamp

<br/>

## Task Description
Instead of identifying bias at the level of individual articles, we focus on sentence-level media bias analysis to identify bias sentences that provide the supportive or background information to shift readers' opinion in an ideological direction, though that may be done via selective inclusion or omission as well as overt ideological language. Specifically, the model takes a whole news article consisting of N sentences as input, and outputs the prediction for each sentence whether it contains bias or not.

<br/>

## Data Description
The sentence-level bias detection task has a relatively short research history. BASIL [1] and BiasdSents [2] dataset are the two available datasets till now that annotate bias sentences with context considered within a news article.
* **BASIL** is the first work to annotate the sentence-level bias and contains 100 triples of articles, each triple consists of three articles from three different media outlets discussing the same event, a total number of 300 articles. We saved the BASIL dataset after processing in the ./BASIL folder. The Cohen's kappa between each annotator and the gold standard is from 0.34 to 0.70.
* **BiasedSents** contains 46 articles with crowd-sourcing annotations in four scales: not biased, slightly biased, biased, and very biased. Following the same scenario of binary judgements [2], we also considered the first two scales as unbiased and the latter two as biased. We used the majority votes to derive the final gold labels from the five different annotators and saved in the ./BiasedSents folder. The Cohen's kappa between each annotator and the gold standard is from 0.17 to 0.58.

<br/>

## Teacher Models
We saved the teacher model for the global functional discourse structure in the ./Teachers/Teacher_global folder, and for the local rhetorical discourse relations in the ./Teachers/Teacher_local folder
* **Teacher_global - global functional discourse structure**
  * **Data:** News Discourse Data [3] is used to train the teacher for global functional discourse structure. (https://github.com/prafulla77/Discourse_Profiling)
  * **Model:** We used the state-of-art model [4] for news discourse profiling as our teacher model. (https://github.com/prafulla77/Discoure_Profiling_RL_EMNLP21Findings)
* **Teacher_local - local rhetorical discourse relations**
  * **Data:** PDTB 2.0 data [5] is used to train the teacher for local rhetorical discourse relations. Followed the official suggestion in PDTB 2.0 dataset, sections 2-21, sections 22 & 24, and section 23 are used for training, development and testing respectively. We saved the PDTB 2.0 data after processing into the folder ./Teachers/Teacher_local/PDTB2_data
  * **Model:** teacher_local_comparison.py is the code to train the local comparison relation teacher, teacher_local_contingency.py is the code to train the local contingency relation teacher. 

<br/>

## Citation
If you are going to cite this paper, please use the form:

Yuanyuan Lei, Ruihong Huang, Lu Wang, and Nick Beauchamp. 2022. Sentence-level Media Bias Analysis Informed by Discourse Structures. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10040–10050, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

<br/>

@inproceedings{lei-etal-2022-sentence,
    title = "Sentence-level Media Bias Analysis Informed by Discourse Structures",
    author = "Lei, Yuanyuan  and
      Huang, Ruihong  and
      Wang, Lu  and
      Beauchamp, Nick",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.682",
    pages = "10040--10050",
    abstract = "As polarization continues to rise among both the public and the news media, increasing attention has been devoted to detecting media bias. Most recent work in the NLP community, however, identify bias at the level of individual articles. However, each article itself comprises multiple sentences, which vary in their ideological bias. In this paper, we aim to identify sentences within an article that can illuminate and explain the overall bias of the entire article. We show that understanding the discourse role of a sentence in telling a news story, as well as its relation with nearby sentences, can reveal the ideological leanings of an author even when the sentence itself appears merely neutral. In particular, we consider using a functional news discourse structure and PDTB discourse relations to inform bias sentence identification, and distill the auxiliary knowledge from the two types of discourse structure into our bias sentence identification system. Experimental results on benchmark datasets show that incorporating both the global functional discourse structure and local rhetorical discourse relations can effectively increase the recall of bias sentence identification by 8.27{\%} - 8.62{\%}, as well as increase the precision by 2.82{\%} - 3.48{\%}.",
}

<br/>

## Reference
[1] Lisa Fan, Marshall White, Eva Sharma, Ruisi Su, Prafulla Kumar Choubey, Ruihong Huang, and Lu Wang. 2019. In Plain Sight: Media Bias Through the Lens of Factual Reporting. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6343–6349, Hong Kong, China. Association for Computational Linguistics.

[2] Sora Lim, Adam Jatowt, Michael Färber, and Masatoshi Yoshikawa. 2020. Annotating and Analyzing Biased Sentences in News Articles using Crowdsourcing. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1478–1484, Marseille, France. European Language Resources Association.

[3] Prafulla Kumar Choubey, Aaron Lee, Ruihong Huang, and LuWang. 2020. Discourse as a function of event: Profiling discourse structure in news articles around the main event. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5374–5386, Online. Association for Computational Linguistics

[4] Prafulla Kumar Choubey and Ruihong Huang. 2021. Profiling news discourse structure using explicit subtopic structures guided critics. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 1594–1605, Punta Cana, Dominican Republic. Association for Computational Linguistics.

[5] Rashmi Prasad, Nikhil Dinesh, Alan Lee, Eleni Miltsakaki, Livio Robaldo, Aravind K. Joshi, and Bonnie LynnWebber. 2008. The penn discourse treebank 2.0. In LREC.
