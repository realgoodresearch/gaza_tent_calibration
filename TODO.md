# TODO

## Validation

- [x] Incorporate actual post-processed values into validation script

- [x] Do random validation (with sufficient samples) by:
    - [x] Municipal region
    - [x] Land use classification (prewar)
    - [x] Time of day
    - [x] Local tent density
    - [x] By spatial context
    - [x] By satellite/sun azimuth/elevation


## Results

- [x] Re-run prediction pipeline for all tiles
    - [x] Re-run processing
    - [x] Re-run predictions
    - [x] Re-run merging

- [x] Do statistical corrections for predictions
    - [x] Re-export h5 validation file
    - [x] Manually validate predictions
    - [x] Validate using cross-validation
        - [x] Replace missing week predictions
        - [x] Verify difference between missing week and missing date
        - [x] Bin time predictions for print and calcualte as distance to/from sunrise
        - [x] Manually spot check random predictions vs. validation data
    - [x] Run negative binomial models
    - [x] Apply corrections to actual predictions (retrain on full training set with publication-level draws/tune)
    - [x] Re-evaluate fixed predictions on hold out set (200 tile hold-out set of validated data)
        - [x] Calculate credible intervals for counts
    - [x] Only test hurdle model
    - [x] Obtain additional high count validation data (200)
    - [x] Apply DBSCAN and yellow line cleaning before applying corrections
    - [x] Run again with high-fidelity MCMC parameters on best model configuration
    - [x] Fix one-hot encoding for categorical variables (e.g. land use, run selection and high-fidelity model selection and count adjustments again)
    - [x] Fix satellite missing metadata in municipality/block count estimations
        - [x] Re-run merge predictions and save satellite data
        - [x] Upload to local folder, ensuring no missing files
        - [x] Write code to incorporate these counts into the municipality/block count pipeline
    - [x] Check that code allows different parameters for hurdle model vs. negative binomial
    - [x] Understand major discrepancies in predictions vs. manual count
    - [ ] Replace corrections with confidence interval?
    - [ ] Calculate error rates by predictor


- [ ] Do data analysis
    - [x] Correct data at municipal and block level, with credible intervals
        [x] Get execution time per week
        [x] Ensure weeks without 80% coverage on that municipality are excluded
    - [x] Re-generate graph for change in displacement using corrected values
    - [ ] Correlate changes in displacement with:
        - [ ] Evac orders
        - [ ] Incidents
    - [ ] Do analysis with corrected count estimates
    - [ ] Brainstorm other analysis ideas
        - [x] Where did tents increase after nearby areas were order to evacuate?
                o Identify blocks receiving an evacuation order nearby in week t, and test whether their tent counts increased in t and t+1.
        - [x] How large and how persistent was displacement following an evacuation order?
                o Compare blocks receiving orders with comparable blocks not receiving orders, measure tent decline immediately after the order and whether counts subsequently recover.
        - [ ] Did areas receiving displaced populations subsequently experience greater military activity?
                o Define an inferred inflow from #1 then model P(Strikeb,t+1​=1)=f(inflowbt​,Tbt​,block FE,week FE) using \(t+1\), not the same week
        - [ ] Map of “number of displacement episodes” plus a distribution of episode counts
        - [ ] Time series of concentration with vertical markers for major military events
                o Calculate each week the concentration of tents across blocks, e.g. tents in top 10% of all tents/all tents, then ask whether concentration rises after periods of intense strikes, evacuation orders or destruction
        - [ ] Correlation between density 
        - [ ] Back up claims in Julia's report with tent data (new chat)
            [ ] Quantify impact of displacement orders on displacement
            [ ] Quantify concentration of displacement into humanitarian zone
            [ ] Quantify number of airstrikes and attacks on civilians in or near dense refugee settlements
            [ ] Quantify increased probability of airstrikes on areas where civilians were told to flee following evacuation orders
            [ ] Quantify clearance of refugee camps based on tank movements and berms

## Writing
- [x] Write data sources section
- [x] Write data analysis section
- [x] Write discussion
- [x] Complete abstract and discussion with information now available from results
- [x] Add Planet SkySat reference, blocks reference
- [x] Mention additional information
    - [x] Data was used to support OCHA operations (where?)
    - [x] We did not account for cloud cover, so tents disappear with high cloud coverage if there are not sufficient images to capture days with no clouds within the week range
    - [x] In discussion section talk about importance of quantifying uncertainty when working with machine learning models (otherwise results cannot be trusted)
    - [x] Number of people in tents, OCHA estimate, implications for degree of displacement in terms of number of people
    - [x] Ensure I mention DBSCAN and yellow line removals
- [ ] Final look over
    - [x] Check correctness of all numbers, ensure DBSCAN/corrected values are being used for all analysis
    - [x] Formatting of data sources and timeline
    - [x] Methodological issues, missing information on methodology (only info that is necessary)
        [ ] Cut machine learning section, move extra to appendix
        [x] Cut data sources section while maintatiing citation
        [ ] Re-add brief methodology for analysis
    - [ ] Spelling, grammar, awkward sentences
    - [ ] Meaningless sentences, unsupported claims
    - [ ] Improve clarify of jargon, plot text
    - [ ] Consiseness, remove unnecessary parts
    - [ ] Connectivity (ensure sections connect, reference and build on each other, ensure figures and tables are referenced)
    - [ ] Missing references


### Model Selection
  PYTENSOR_FLAGS='base_compiledir=/tmp/pytensor' \
  MPLCONFIGDIR=.mplconfig \
  python src/calibration/pipeline.py \
    --seed 2026 \
    --selection-mode run \
    --cv-model-kind hurdle_nb \
    --cv-split random \
    --cv-folds 3 \
    --cv-repeats 1 \
    --cv-draws 120 \
    --cv-tune 120 \
    --cv-chains 2 \
    --target-accept 0.97 \
    --min-fold-win-rate 0.66 \
    --mae-epsilon 0.001 \
    --skip-full-application \
    --allow-poor-mcmc


  PYTENSOR_FLAGS='base_compiledir=/tmp/pytensor'
  MPLCONFIGDIR=.mplconfig python src/calibration/pipeline.py
  --seed 2026 --uncertainty-features --uncertainty-mean-sigma
  0.15 --selection-mode run --pub-draws 2000 --pub-tune 2000
  --pub-chains 4 --pub-target-accept 0.995 --apply-level both
  --municipality-coverage-threshold 0.8 --no-resume-
  application

### Final Model Fit

  PYTENSOR_FLAGS='base_compiledir=/tmp/pytensor' \
  MPLCONFIGDIR=.mplconfig \
  python src/calibration/pipeline.py \
    --seed 2026 \
    --selection-mode use-best \
    --best-spec-path src/calibration/artifacts/selected_model.json \
    --pub-draws 2000 \
    --pub-tune 2000 \
    --pub-chains 4 \
    --pub-target-accept 0.995 \
    --apply-level both \
    --municipality-coverage-threshold 0.8 \
    --no-resume-application

  PYTENSOR_FLAGS='base_compiledir=/tmp/pytensor'
  MPLCONFIGDIR=.mplconfig python src/calibration/pipeline.py
  --seed 2026 --selection-mode use-best --best-spec-path src/
  calibration/artifacts/selected_model.json --pub-draws 2000
  --pub-tune 2000 --pub-chains 4 --pub-target-accept 0.995
  --apply-level both --municipality-coverage-threshold 0.8
  --no-resume-application



  MPLCONFIGDIR=.mplconfig python src/validation/manual_validation/sanity_check.py /
  --csv-path outputs/validation/manual_validation_results_with_raw_and_postprocessed_predictions.csv /
  --prediction-column prediction_count_postprocessed