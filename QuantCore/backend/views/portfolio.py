from django.shortcuts import render
from rest_framework.views import APIView

class PortfolioAPIView(APIView):
    def get(self, request):
        return render(request, 'portfolio.html')
