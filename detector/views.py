from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import URLForm
from ml.inference import predict_url

def index(request):
    form = URLForm()
    return render(request, "detector/index.html", {"form": form})

def predict(request):
    result = None
    error = None
    if request.method == "POST":
        form = URLForm(request.POST)
        if form.is_valid():
            print("Form is valid!")
            url = form.cleaned_data["url"]
            try:
                result = predict_url(url)
                print(f"Prediction Result: {result}") # Debugging line
            except Exception as e:
                import traceback
                traceback.print_exc()
                error = str(e)
        else:
            print(f"Form is NOT valid. Errors: {form.errors}")
    else:
        form = URLForm()
    return render(request, "detector/index.html", {"form": form, "result": result, "error": error})
