%% MLP Neural Network 
% 2 hidden layers
% Comments in Polish
% Author: Pawe³ Mêdyk

% Wielowarstwowa sieæ neuronowa MLP
% 2 warstwy ukryte
% Imiê i nazwisko autora: Pawe³ Mêdyk

close all
skala = 3;
%% konstruowanie bazy uczacej
baza_ucz_we=1:10;   % dane ucz. wejsciowe
baza_ucz_wy=sin(baza_ucz_we/4)+cos(baza_ucz_we); % dane ucz.wyjsciowe
baza_ucz_wy=baza_ucz_wy/skala; % skalowanie danych ucz.wej.

%% konstruowanie bazy testowej
baza_test_we=[1:0.1:10]; % dane test.wej.
baza_test_wy=sin(baza_test_we/4)+cos(baza_test_we); % dane test.wyj.
baza_test_wy=baza_test_wy/skala; % skalowanie danych test.wyj.

%% struktura sieci
n = size(baza_ucz_we,1);  % liczba neronow/wejsc w warstwie wejsciowej
k1 = 6;                    % liczba neuronow w pierwszej warstwie ukrytej
k2 = 6;                    % liczba neuronow w drugiej warstwie ukrytej
k3 = size(baza_ucz_wy,1);   % liczba neuronow w warstwie wyjsciowej

%% parametry sieci
eta1 = 0.15; % wspolczynnik uczenia sie I warstwy ukrytej 
eta2 = 0.15; % wspolczynnik uczenia sie II warstwy ukrytej
eta3 = 0.05; % wspolczynnik uczenia sie warstwy wyj
beta1 = 1.25;    % wspolczynnik stromosci funkcji aktywacji I w.ukr.
beta2 = 1.25;    % wspolczynnik stromosci funkcji aktywacji II w.ukr.
beta3 = 1.5;     % wspolczynnik stromosci funkcji aktywacji w.wyj.
Epoki = 200000;  % liczba epok

%% liczba instancji
lowest_MSE=10^5;% inicjalizuj najmniejszy MSE duza wartoscia
N=5             % liczba iteracji/powtorzen eksperymentu
%% petla glowna - generuj nowa instancje sieci N razy
for i=1:N
%% inicjowanie macierzy wag
a = -0.5; % dolna granica inicjowania wag
b = 0.5;  % górna granica inicjowania wag
W1 = (b-a)*rand(n+1,k1)+a; % macierz wag dla pierwszej warstwy ukrytej
W2 = (b-a)*rand(k1+1,k2)+a; % macierz wag dla warstwy wyjsciowej
W3 = (b-a)*rand(k2+1,k3)+a; % macierz wag dla warstwy wyjsciowej

%% petla uczaca
for ep = 1: Epoki
   L = randi([1, size(baza_ucz_we,2)],1);   % losuj zestaw uczacy
   y0=baza_ucz_we(:,L);             % wyjscie warstwy wejsciowej
   
   x1 = [-1; y0];                   % zbuduj wektor wejsc dla I w.ukrytej (dodanie wej. bias)
   u1=W1'*x1;                       % wyznacz wektor pobudzen I warstwy ukrytej
   y1=1./(1+exp(-beta1*u1));        % wyjscie I warstwy ukrytej; 
                                % funkcja aktywacji I warstwy ukrytej - sigmoidalna unipolarna
   x2 = [-1; y1];                   % zbuduj wektor wejsc dla II w.ukrytej (dodanie wej. bias)
   u2=W2'*x2;                       % wyznacz wektor pobudzen II warstwy ukrytej
   y2=1./(1+exp(-beta2*u2));        % wyjscie II warstwy ukrytej; 
                                % funkcja aktywacji II warstwy ukrytej - sigmoidalna unipolarna
   x3=[-1; y2];                     % zbuduj wektor wejsc dla warstwy wyjsciowej (dodanie wej. bias)
   u3=W3'*x3;                     	% wyznacz wektor pobudzen warstwy wyjsciowej
   y3=2*1./(1+exp(-beta3*u3))-1;    % wyjscie warstwy wyjsciowej; 
                                % funkcja aktywacji warstwy wyjsciowej - sigmoidalna bipolarna
   
   ty = baza_ucz_wy(:,L);           % oczekiwany wektor wyjscia dla wektora uczacego
   d3=ty-y3;                        % wektor bledu pomiedzy wyjsciem sieci, a wyjsciem oczekiwanym
   df3=beta3/2*(1-y3*y3);           % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -w.wyj
   dW3=eta3*x3*(d3.*df3)';          % macierz poprawek dla warstwy wyjsciowej
   W3=W3+dW3;                       % aktualizacja macierzy wag warstwy wyjsciowej

   d2=W3*d3;                        % obliczanie wspolczynnika bledu; wektor sum (blad warstwy wyzszej)*waga
   df2=x3.*(1-x3);                  % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -II w.ukryta
   dd2=(d2.*df2)';                  % wektor wspolczynnikow bledu 
   dd2=dd2(2:end);                  % skrocenie wektora wspolczynnikow bledu
   dW2=eta2*x2*dd2;                 % macierz poprawek dla II w.ukrytej; wsteczna propagacja bledu
   W2=W2+dW2;
   
   d1=W2*d2(2:end);                 % obliczanie wspolczynnika bledu; wektor sum (blad warstwy wyzszej)*waga
   df1=x2.*(1-x2);                  % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -I w.ukryta
   dd1=(d1.*df1)';                  % wektor wspolczynnikow bledu
   dd1=dd1(2:end);                  % skrocenie wektora wspolczynnikow bledu 
                                % nie mozna poprawiac wektora wag dla
                                % wejscia bias, bo ma ono stala wartosc
   dW1=eta1*x1*dd1;                 % macierz poprawek dla warstwy wyjsciowej; wsteczna propagacja bledu
   W1=W1+dW1;                       % aktualizacja macierzy wag warstwy ukrytej
    
end

%% test instancji sieci
Y_matrix=baza_test_wy;
for przykl = 1 : size(baza_test_we,2) % dla wszystkich przykladow z bazy ucz.
    x1= [-1; baza_test_we(:,przykl)];   % wejscie w.ukrytej
    u1=W1'*x1;                          % pobudzenie w.ukrytej
    y1=1./(1+exp(-beta1*u1));           % wyjscie w.ukrytej    
    x2=[-1; y1];                        % wejscie w.wyjsciowej
    u2=W2'*x2;                          % pobudzenie w.wyjsciowej
    y2=1./(1+exp(-beta2*u2));           % wyjscie w.wyjsciowej
    x3=[-1; y2];                        % wejscie w.wyjsciowej
    u3=W3'*x3;                          % pobudzenie w.wyjsciowej
    y3=2*1./(1+exp(-beta3*u3))-1;       % wyjscie w.wyjsciowej
    
    Y_matrix(:,przykl)=y3;              % zapisz wynik dla tego przykladu w macierzy
end

MSE=immse(Y_matrix,baza_test_wy);
%% zapis najlepszego rozwiazania
% jesli MSE mniejsze niz najmniejsze znane, zapamietaj macierze wag
    if MSE < lowest_MSE
        lowest_MSE=MSE
        W1_best=W1;
        W2_best=W2;
        W3_best=W3;
    end
end


%% prezentacja najlepszego wypracowanego rozwiazania
% najmniejszy MSE
W1=W1_best;
W2=W2_best;
W3=W3_best;

% odp. na zestaw testowy
Y_matrix=baza_test_wy;
for przykl = 1 : size(baza_test_we,2) % dla wszystkich przykladow z bazy ucz.
    x1= [-1; baza_test_we(:,przykl)];   % wejscie w.ukrytej
    u1=W1'*x1;                          % pobudzenie w.ukrytej
    y1=1./(1+exp(-beta1*u1));           % wyjscie w.ukrytej    
    x2=[-1; y1];                        % wejscie w.wyjsciowej
    u2=W2'*x2;                          % pobudzenie w.wyjsciowej
    y2=1./(1+exp(-beta2*u2));           % wyjscie w.wyjsciowej
    x3=[-1; y2];                        % wejscie w.wyjsciowej
    u3=W3'*x3;                          % pobudzenie w.wyjsciowej
    y3=2*1./(1+exp(-beta3*u3))-1;       % wyjscie w.wyjsciowej
    
    Y_matrix(:,przykl)=y3;              % zapisz wynik dla tego przykladu w macierzy
end
MSE=immse(Y_matrix,baza_test_wy)

%%  wykres - najlepsze wypracowane rozwiazanie
figure(2);
hold on
plot(baza_test_we,Y_matrix)     % odpowiedz sieci na zestaw testowy
plot(baza_test_we,baza_test_wy) % zestaw testowy
plot(baza_ucz_we,baza_ucz_wy,'*')   % zestaw uczacy
legend('odp.sieci','odp.oczekiwana', 'odp.zest.ucz.')
xlabel('x')
ylabel('y')


