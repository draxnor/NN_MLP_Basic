%% MLP Neural Network 
% 1 hidden layer
% Comments in Polish
% Author: Pawe³ Mêdyk

% Wielowarstwowa sieæ neuronowa MLP
% 1 warstwa ukryta
% Imiê i nazwisko autora: Pawe³ Mêdyk

%close all
skala = 3;  % odwrotnosc skali dla wek.wyjscia z bazy uczacej i testowej
%% konstruowanie bazy uczacej
baza_ucz_we=1:10;   % baza ucz. - wejscia
baza_ucz_wy=sin(baza_ucz_we/4)+cos(baza_ucz_we);    % baza ucz. - wyjscia  
baza_ucz_wy=baza_ucz_wy/skala; % skalowanie aproksymowanej funkcji (b.ucz.)

%% konstruowanie bazy testowej
baza_test_we=1:0.1:10;  % baza test. - wejscia
baza_test_wy=sin(baza_test_we/4)+cos(baza_test_we); % baza test. - wyjscia
baza_test_wy=baza_test_wy/skala; % skalowanie aproksymowanej funkcji (b.test.)

%% struktura sieci
n = size(baza_ucz_we,1);  % liczba wejsc/neronow w warstwie wejsciowej
k1 = 10;                  % liczba neuronow w pierwszej warstwie ukrytej
k2 = size(baza_ucz_wy,1); % liczba neuronow w warstwie wyjsciowej

%% petla porownawcza
indeks_wsp=2; % indeks sprawdzanego wspolczynnika; eta/beta/Epoki etc. nalezy zamienic w kodzie
T=1; % ilosc porownan / liczba sprawdzanych wartosci danego parametru
MSE_porownaj=zeros(T,1);

for j=1:T
%% parametry sieci
eta= zeros(2,1);
beta=zeros(2,1);
eta(1) = 0.15;    % wspolczynnik uczenia sie warstwy ukrytej
eta(2) = 0.05;    % wspolczynnik uczenia sie warstwy wyj.
beta(1) = 1.15;   % wspolczynnik stromosci funkcji aktywacji - warstwa ukryta I
beta(2) = 1.5;    % wsp. stromosci f.akt. - warstwa wyj.    
Epoki = 200000;   % liczba epok

% wartosci do testowania wplywu parametrow na jakosc rozwiazania
eta_vec = [0.05, 0.15, 0.35];
beta_vec= [1.05, 1.35, 1.80];
Epoki_vec=[50000 200000 500000];

%eta(indeks_wsp)=eta_vec(j);
%beta(indeks_wsp)=beta_vec(j);     % test wplywu wspolczynnika
%Epoki = Epoki_vec(j);

%% liczba instancji
lowest_MSE=10^6; % inicjalizuj najmniejszy MSE duza wartoscia
N=5  % liczba iteracji/powtorzen eksperymentu; N instancji sieci
%% petla glowna - generuj nowa instancje sieci N razy
for i=1:N
%% inicjowanie macierzy wag
a = -0.5; % dolna granica inicjowania wag
b = 0.5;  % górna granica inicjowania wag
W1 = (b-a)*rand(n+1,k1)+a; % macierz wag dla pierwszej warstwy ukrytej
W2 = (b-a)*rand(k1+1,k2)+a;% macierz wag dla warstwy wyjsciowej

%% petla uczaca
for ep = 1: Epoki
   L = randi([1, size(baza_ucz_we,2)],1);   % losuj zestaw uczacy
   y0=baza_ucz_we(:,L);             % wyjscie warstwy wejsciowej
   x1 = [-1; y0];                   % zbuduj wektor wejsc I w.ukrytej (dodanie wej. bias)
   u1=W1'*x1;                       % wyznacz wektor pobudzen pierwszej warstwy ukrytej
   y1=1./(1+exp(-beta(1)*u1));      % wyjscie I warstwy ukrytej;
                % funkcja aktywacji pierwszej warstwy ukrytej - sigmoidalna unipolarna
   
   x2=[-1; y1];                     % zbuduj wektor wejsc dla warstwy wyjsciowej (dodanie wej. bias)
   u2=W2'*x2;                     	% wyznacz wektor pobudzen warstwy wyjsciowej
   y2=2*1./(1+exp(-beta(2)*u2))-1;  % wyjscie warstwy wyjsciowej; 
                % funkcja aktywacji warstwy wyjsciowej - sigmoidalnabipolarna
   
   ty = baza_ucz_wy(:,L);           % oczekiwany wektor wyjscia dla wektora uczacego
   d2=ty-y2;                        % wektor bledu pomiedzy wyjsciem sieci, a wyjsciem oczekiwanym
   df2=beta(2)/2*(1-y2*y2);         % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -w.wyj
   dW2=eta(2)*x2*(d2.*df2)';        % macierz poprawek dla warstwy wyjsciowej
   W2=W2+dW2;                       % aktualizacja macierzy wag warstwy wyjsciowej

   d1=W2*d2;                % obliczanie wspolczynnika bledu; wektor sum (blad warstwy wyzszej)*waga
   df1=x2.*(1-x2);          % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -w.ukryta
   dd1=(d1.*df1)';          % wektor wspolczynnikow bledu
   dd1=dd1(2:end);          % skrocenie wektora wspolczynnikow bledu 
                          % nie mozna poprawiac wektora wag dla
                          % wejscia bias, bo ma ono stala wartosc
   dW1=eta(1)*x1*dd1;       % macierz poprawek dla warstwy wyjsciowej; wsteczna propagacja bledu
   W1=W1+dW1;               % aktualizacja macierzy wag warstwy ukrytej
    
end

%% test instancji sieci
Y_matrix=baza_test_wy;
for przykl = 1 : size(baza_test_we,2) % dla wszystkich przykladow z bazy ucz.
    x1= [-1; baza_test_we(:,przykl)];   % wejscie w.ukrytej
    u1=W1'*x1;                          % pobudzenie w.ukrytej
    y1=1./(1+exp(-beta(1)*u1));         % wyjscie w.ukrytej    
    x2=[-1; y1];                        % wejscie w.wyjsciowej
    u2=W2'*x2;                          % pobudzenie w.wyjsciowej
    y2=2*1./(1+exp(-beta(2)*u2))-1;     % wyjscie w.wyjsciowej
    Y_matrix(:,przykl)=y2;              % zapisz wynik dla tego przykladu w macierzy wyników
end
MSE=immse(Y_matrix,baza_test_wy);

%% zapis najlepszej macierzy wag
% jesli MSE mniejsze niz najmniejsze znane, zapamietaj macierze wag
    if MSE < lowest_MSE
        lowest_MSE=MSE;
        W1_best=W1;
        W2_best=W2;
    end
end


%% prezentacja najlepszego wypracowanego rozwiazania
% najmniejszy MSE
W1=W1_best;
W2=W2_best;
Y_matrix=baza_test_wy;
for przykl = 1 : size(baza_test_we,2) % dla wszystkich przykladow z bazy ucz.
    x1= [-1; baza_test_we(:,przykl)];   % wejscie I w.ukrytej
    u1=W1'*x1;                          % pobudzenie I w.ukrytej
    y1=1./(1+exp(-beta(1)*u1));         % wyjscie I w.ukrytej    
    x2=[-1; y1];                        % wejscie w.wyjsciowej
    u2=W2'*x2;                          % pobudzenie w.wyjsciowej
    y2=2*1./(1+exp(-beta(2)*u2))-1;     % wyjscie w.wyjsciowej
    Y_matrix(:,przykl)=y2;              % zapisz wynik dla tego przykladu w macierzy wyników
end
MSE=immse(Y_matrix,baza_test_wy)

x=floor(log10(MSE));
MSE_porownaj(j)=MSE; % zapamietaj najmniejsza uzyskana wartosc MSE dla danej wartosci parametru
%%  wykres - najlepsze wypracowane rozwiazanie
figure(1);
subplot(1,T,j) % do porownania wplywu liczby warstw
hold on
plot(baza_test_we,Y_matrix)     % odpowiedz sieci na zestaw testowy
plot(baza_test_we,baza_test_wy) % zestaw testowy
plot(baza_ucz_we,baza_ucz_wy,'*')   % zestaw uczacy
%title_str= sprintf('Epoki=%d; MSE=%.2f*10^{%d})', Epoki,MSE/(10^x),x)
%title_str= sprintf('beta%d=%.2f; MSE=%.2f*10^{%d})',indeks_wsp, beta(indeks_wsp),MSE/(10^x),x)
%title_str= sprintf('eta%d=%.2f; MSE=%.2f*10^{%d})',indeks_wsp, eta(indeks_wsp),MSE/(10^x),x)
title_str= sprintf('1 warstwa ukryta; MSE=%.2f*10^{%d}',MSE/(10^x),x)
title(title_str)
legend('odp.sieci','w. aproksymacji', 'odp.zest.ucz.','Location','northwest')
xlabel('x')
ylabel('y')

end