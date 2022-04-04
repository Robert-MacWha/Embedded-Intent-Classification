# Embedded Classification
An embedding-based intent classification system that was initially developed for Aurras v2.  It is now partially depreciated since the project has shifted to an intent-less system.  However, the embedding systems will be used to extract entities and possible actions in Aurras V2.

## Intent classification
![Sample image](https://github.com/Robert-MacWha/Embedded-Intent-Classification/blob/main/sample.png)
_Known intents were included in the dataset, unknown intents were not included in the dataset.  Note how the model is still able to relate and disassociate these intents that it has never seen_

The embedded intent classification works by embedding an input prompt (IE What's my schedule for today) into a latent space where its position is encoded based on the intent of the user.  Similar intents will be in similar locations.  This means that we can group the intents and extract them based on proximity.  Furthermore, novel intents can be added without model re-training since, if sufficiently intelligent, the model will know to place these new intents at new positions.
