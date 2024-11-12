VENV_NAME = asl_words_interpreter

.PHONY: setup
setup:
	python3 -m venv $(VENV_NAME)
	./$(VENV_NAME)/bin/pip install -r requirements.txt

.PHONY: activate
activate:
	@echo "To activate the virtual environment:"
	@echo "For macOS/Linux: source $(VENV_NAME)/bin/activate"
	@echo "For Windows: .\\$(VENV_NAME)\\Scripts\\activate"

.PHONY: clean
clean:
	rm -rf $(VENV_NAME)

.PHONY: install
install: setup activate
	@echo "Installation complete. Virtual environment '$(VENV_NAME)' is ready."
