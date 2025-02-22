# Deploying to Railway

This guide explains how to deploy the Kokoro TTS API to Railway.

## Prerequisites

1. A Railway account
2. Railway CLI installed (optional)

## Deployment Steps

1. Fork this repository to your GitHub account

2. Create a new project in Railway:
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your forked repository

3. Configure Environment Variables:
   - In your Railway project, go to the "Variables" tab
   - Add the following required environment variables:
     ```
     USE_GPU=false
     OUTPUT_DIR=/tmp/output
     TEMP_FILE_DIR=/tmp/kokoro_temp
     MODEL_DIR=/app/api/src/models
     VOICES_DIR=/app/api/src/voices/v1_0
     ```

4. Deploy:
   - Railway will automatically deploy your application
   - The deployment process may take a few minutes

## Important Notes

1. The free tier of Railway has the following limitations:
   - 512 MB RAM
   - 1 GB disk space
   - No GPU support

2. Model and voice files:
   - Ensure your model and voice files are within Railway's storage limits
   - Consider using cloud storage for larger files

3. Temporary files:
   - The application uses `/tmp` directory for temporary files
   - Files are automatically cleaned up based on the configured settings

## Monitoring

1. Check the deployment status in Railway dashboard
2. Monitor logs for any issues
3. Use the `/health` endpoint to check API status

## Troubleshooting

1. If deployment fails:
   - Check the logs in Railway dashboard
   - Ensure all required environment variables are set
   - Verify the Python version (3.10+) is supported

2. If the API is slow:
   - Consider upgrading to a paid tier
   - Optimize model loading and caching
   - Adjust temporary file cleanup settings

## Support

For issues related to:
- Railway deployment: [Railway Support](https://railway.app/support)
- Kokoro TTS API: Open an issue in the GitHub repository 