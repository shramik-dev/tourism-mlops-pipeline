
name: MLOps Pipeline

on:
  push:
    branches:
      - main
      - feature/*
  pull_request:
    branches:
      - main

env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
  PYTHON_VERSION: '3.12'

jobs:
  ml-pipeline:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      checks: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Create data directory
        run: mkdir -p data

      - name: Data Preparation
        run: python src/data_prep.py
        env:
          OUTPUT_DIR: ./data

      - name: Verify Data File
        run: |
          if [ ! -f data/processed.csv ]; then
            echo "Error: data/processed.csv not found!"
            exit 1
          fi
          echo "Data file exists. Content preview:"
          head -n 5 data/processed.csv

      - name: Model Training
        run: python src/train.py
        env:
          MODEL_OUTPUT: ./models/model.joblib
          DATA_PATH: ./data/processed.csv

      - name: Model Evaluation
        id: eval
        run: python src/evaluate.py
        env:
          MODEL_PATH: ./models/model.joblib
          TEST_DATA: ./data/test.csv
        continue-on-error: false

      - name: Run Tests
        run: pytest tests/ --verbose
        env:
          MODEL_PATH: ./models/model.joblib

      - name: Upload Model Artifact
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: ./models/model.joblib

      - name: Deploy to Hugging Face Spaces (on main branch only)
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: |
          pip install huggingface_hub
          huggingface-cli login --token ${{ env.HF_TOKEN }}
          huggingface-cli upload your-username/tourism-rf-model ./models/model.joblib model.joblib
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}

      - name: Log Metrics to PR/Run Summary
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            let eval_output = 'No evaluation results available';
            try {
              eval_output = fs.readFileSync('./evaluation_results.json', 'utf8');
            } catch (error) {
              if (error.code === 'ENOENT') {
                console.warn('evaluation_results.json not found, using default message.');
              } else {
                throw error;
              }
            }
            const summary = `## ML Pipeline Results\n${eval_output}`;
            github.rest.checks.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              name: 'MLOps Pipeline',
              head_sha: context.sha,
              status: 'completed',
              conclusion: '${{ job.status }}',
              output: { title: 'Evaluation Metrics', summary }
            });

      - name: Auto-Push to Main (from feature branches)
        if: github.ref != 'refs/heads/main' && success()
        uses: actions/github-script@v7
        with:
          script: |
            try {
              const { data: { default_branch } } = await github.rest.repos.get(context.repo);
              if (default_branch === 'main') {
                const branchName = context.ref.replace('refs/heads/', '');
                const pr = await github.rest.pulls.create({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  title: `Auto-merge from ${branchName}`,
                  head: branchName,
                  base: 'main'
                });
                console.log(`Created PR #${pr.data.number} for ${branchName} into main.`);
                // Note: Approval and merge are disabled due to permissions restriction
                // await github.rest.pulls.createReview({...});
                // await github.rest.pulls.merge({...});
              }
            } catch (error) {
              console.error('Error in auto-push:', error.message);
              if (error.status === 403 && error.message.includes('not permitted to create or approve pull requests')) {
                console.warn('GitHub Actions lacks permission to create/approve PRs. Please update repository settings or use a PAT.');
              } else {
                throw error;
              }
            }
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  