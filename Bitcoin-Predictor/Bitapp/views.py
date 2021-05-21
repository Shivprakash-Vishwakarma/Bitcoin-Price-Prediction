from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from .forms import SignUpForm
from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from nsepy import get_history
from Bitapp.models import Companies
from django.contrib import messages
from Bitcoin.run import RunModel
from datetime import date, timedelta
from GoogleNews import GoogleNews
import pandas as pd


def home(request):
    # news data
    googlenews = GoogleNews(lang='en', period='12h', encode='utf-8')
    googlenews.get_news('Cryptocurrency')
    news = googlenews.result(sort=True)
    news_first = news[:8]
    news_length = []
    for i in range(1, len(news)//8):
        news_length.append(news[i*8:(i*8)+8])

    # chart data
    # companies = Companies.objects.all()
    # data = get_history(
    #     symbol="NIFTY",
    #     start=date.today() - timedelta(days=30),
    #     end=date.today(),
    #     index=True,
    # )

    bitstamp = pd.read_csv(
        r"C:\Users\afzal\Desktop\Bitcoin-Predictor\Bitcoin-Predictor\Bitcoin\trained_models\bitt.csv")

    # price_series = bitstamp.reset_index().Close.values

    from sklearn.preprocessing import MinMaxScaler
    data = bitstamp['Close']
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit_transform(price_series.reshape(-1, 1))
    labels = ' $ BITCOIN PRICE PREDICTOR $ '
    context = {
        'labels': labels,
        'data': data.tolist(),
        'news': news,
        'news_length': news_length,
    }

    if request.POST.get('login'):
        user = authenticate(
            request, username=request.POST['username'], password=request.POST.get('password'))

        if user is not None:
            login(request, user)
            return redirect('home')

        else:
            print('Welcome '+user)
            context['error'] = "*Username and Password doesn't Match.*"
            # context['results']=results

            return render(request, 'home.html', context=context)

    if request.POST.get('option'):

        pk = request.POST['option']
        print(pk, flush=True)

        obj = RunModel()

        current_data = data
        # print(current_data)

        current_labels = bitstamp.reset_index().Timestamp.values.tolist()
        nan_ = [float('nan') for i in range(348)]
        # nan_.append(current_data.tolist()[-1])
        # priceObj = obj.getPrice()
        # d = [[NaN] for i in range(328)]
        nextDays = obj.getNextQDays(pk)

        color = []
        if current_data.tolist()[-1] > nextDays[-1]:
            color.append(True)
        else:
            color.append(False)

        # context['priceObj'] = priceObj
        context['nextDays'] = nextDays
        context['nextDays_data'] = nan_ + nextDays
        context['nextDays_labels'] = current_labels + list(range(1, int(pk)+1))
        # context['selectedOption'] = company.name
        context['current_data'] = data.tolist()
        context['current_labels'] = current_labels
        context['color'] = color
        context['pk'] = pk
        context['onepiece'] = current_data.tolist()[-1]

    return render(request, 'home.html', context=context)


def signupuser(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()

            return redirect('home')
        else:
            messages.warning(request, form.errors)

    return render(request, 'signup.html', {'form': SignUpForm()})


def logoutuser(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home')
