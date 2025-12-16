# Contributing to SkillHub

First off, thank you for considering contributing to SkillHub! It's people like you that make SkillHub such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for SkillHub. Following these guidelines helps maintainers understand your report, reproduce the behavior, and find related reports.

**Before Submitting A Bug Report**
- Check the debugging guide in the documentation
- Check that your issue isn't already filed
- Collect information about the bug

**How Do I Submit A Good Bug Report?**

Bugs are tracked as GitHub issues. Create an issue and provide the following information:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed
- Explain the behavior you expected to see
- Include screenshots and animated GIFs if possible
- Include your environment details

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for SkillHub.

**Before Submitting An Enhancement Suggestion**
- Check that your idea fits with the scope and aims of the project
- Make sure that your suggestion isn't already present
- Read the documentation thoroughly

**How Do I Submit A Good Enhancement Suggestion?**

Enhancement suggestions are tracked as GitHub issues. Create an issue and provide the following information:

- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- List some other applications where this enhancement exists
- Specify which version you're using
- Specify the name and version of the OS you're using

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs in your pull request whenever possible
- Follow the Python styleguide
- Include thoughtfully-worded, well-structured tests
- Document new code based on the Documentation Styleguide
- End all files with a newline

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
    - 🎨 `:art:` when improving the format/structure of the code
    - 🐎 `:racehorse:` when improving performance
    - 📝 `:memo:` when writing docs
    - 🐛 `:bug:` when fixing a bug
    - 🔥 `:fire:` when removing code or files
    - 💚 `:green_heart:` when fixing the CI build
    - ✅ `:white_check_mark:` when adding tests
    - ⬆️ `:arrow_up:` when upgrading dependencies
    - 🔒 `:lock:` when dealing with security

### Python Styleguide

- Follow PEP 8
- Use type hints
- Write docstrings for all public methods
- Keep functions focused and small
- Use meaningful variable names

### Documentation Styleguide

- Use Markdown
- Reference functions with backticks: `my_function()`
- Include code examples when relevant
- Keep line length to 80 characters
- Include section headers
- Link to other docs when appropriate

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests.

#### Type of Issue and Issue State

- `enhancement` - Feature requests
- `bug` - Confirmed bugs or reports likely to be bugs
- `question` - Questions more than bug reports or feature requests
- `feedback` - General feedback
- `help-wanted` - The core team would appreciate help from the community
- `beginner` - Less complex issues which would be good first issues
- `more-information-needed` - More information needs to be collected
- `needs-reproduction` - Likely bugs, but haven't been reliably reproduced
- `blocked` - Issues blocked on other issues
- `duplicate` - Issues which are duplicates of other issues
- `wontfix` - The core team has decided not to fix these issues
- `invalid` - Issues which aren't valid (e.g., user errors)

#### Topic Categories

- `documentation` - Related to any type of documentation
- `performance` - Related to performance
- `security` - Related to security
- `ui` - Related to visual design
- `api` - Related to SkillHub's public APIs
- `uncaught-exception` - Issues about uncaught exceptions
- `crash` - Reports of SkillHub completely crashing
- `network` - Related to network problems or working with remote files
- `git` - Related to Git functionality
- `blocked` - Related to blocked updates

#### Pull Request Labels

- `work-in-progress` - Pull requests which are still being worked on
- `needs-review` - Pull requests which need code review
- `under-review` - Pull requests being reviewed
- `requires-changes` - Pull requests which need to be updated
- `needs-testing` - Pull requests which need manual testing