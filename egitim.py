import torch #Model hesaplarını yapacak kütüphane
import torch.nn as nn #İçindeki model parçaları kutusunu (nn = neural network) alır ve kısaca nn diye çağıracağımızı söyler

metin = "anaconda" #modelin çalışacağı kelime
harfler = sorted(set(metin)) #kelimedeki harflerin alfabetik listesi
harf_numara = {h: i for i, h in enumerate(harfler)} #Her harfe sırayla bir numara veriyoruz alfabetik olarak sıraladıktan sonra 
numara_harf = {i: h for h, i in harf_numara.items()} #Burda da tam tersini yapıyoruz numaraları harfe çeviriyoruz model sonunda numara üretecek biz onu tekrar harfe çevirmek için buna ihtiyaç duyacaz

sayilar = [harf_numara[h] for h in metin] #metindeki her harfin numarasını sözlükten bulup listeye diziyoruz
print(harfler) #Harf listesini ekrana yazdırıyoruz
print(harf_numara) #Hangi harfin hangi numarayı aldığını görmek için sözlüğü ekrana yazdırıyoruz
print(sayilar) #Kelimenin numara halini ekrana yazdırıyoruz

# Tablo oluşturuyoruz: 6 harf x 2 sayı = 12 parametre
tablo = nn.Embedding(len(harfler), 2) #bir tablo yaratıyoruz Bilgisayar tablonun hücrelerini rastgele sayılarla doldurur zar atar gibi
print(tablo.weight) #Tablonun içindeki sayıları ekrana yazdırıyoruz
print(tablo.weight.numel()) #Tablodaki toplam hücre sayısını yazdırıyoruz

# "merhaba"nın numaralarıyla tablodan her harfin kartını çekiyoruz
girdi = torch.tensor(sayilar) # Numara listesini PyTorch'un sayı kutusuna koyar, sayılar değişmez
kartlar = tablo(girdi) # Her numaranın kartını tablodan çeker (kopya gelir, tablo değişmez)
print(kartlar) #tüm kartları yazdırır

# Her kartı “sıradaki harf hangisi olabilir?” sorusuna çevireceğiz. Karttan 6 puan çıkacak, her harf için bir puan. Puanı en yüksek harf, modelin tahmini olacak.
cikis = nn.Linear(2, len(harfler)) #yeni bir tablo daha yapıyoruz 2 sayı gir 6 puan çık diyen bir puanlama makinesi. Bu da rastgele sayılarla dolar, yani modele yeni parametreler eklemiş olduk
skorlar = cikis(kartlar) #7 kartı bu makineye veriyoruz her kart için 6 puan çıkıyor
print(skorlar) #Puanları ekrana yazdırıyor

olasilik = torch.softmax(skorlar, dim=1) #Her satırdaki 6 puanı yüzdeye çevirir. Hepsini toplarsan 1 eder.
print(olasilik[0]) #İlk satırı, yani m harfinden sonra gelecek harf için modelin tahminini yazdırır.


hedef = torch.tensor(sayilar[1:]) #Bu, cevap anahtarı. sayilar[1:] listenin ilk elemanını atar:
kayip = nn.functional.cross_entropy(skorlar[:-1], hedef) #Modelin tahminlerini cevap anahtarıyla karşılaştırıp ceza puanı veriyor.
print(kayip) #Ceza puanını yazdırır.


optimizer = torch.optim.SGD(list(tablo.parameters()) + list(cikis.parameters()), lr=0.5) #Bu satır, modelin 30 ayar düğmesini (iki tablodaki bütün sayıları), her seferinde 0,5 gücünde çevirecek bir “düzeltici” hazırlıyor.
kayip.backward() #Her düğme için “bunu biraz çevirirsem ceza ne kadar değişir?” hesabını yapar. Sonucu her sayının yanına not olarak yazar.
print(tablo.weight.grad) #Kart tablosundaki sayılar için çıkan notları yazdırır. grad, bu notun adı

optimizer.step() #Parametreleri günceller. Her parametre için bir önceki adımda hesapladığımız gradient’e bakar. Gradient eksiyse parametreyi biraz artırır, artıysa biraz azaltır. Değişim miktarı öğrenme hızına (lr=0.5) bağlıdır. Ekrana bir şey yazmaz.
kartlar = tablo(girdi) #“merhaba”nın numaralarını tekrar tabloya verip kartları çekiyoruz. Tablodaki sayılar az önce değiştiği için gelen kartlar eskilerden biraz farklı.
skorlar = cikis(kartlar) #Yeni kartları puanlama katmanına veriyoruz. O da her harf için 6 puan üretiyor. Bu katmanın parametreleri de güncellendiği için puanlar da değişti.
kayip2 = nn.functional.cross_entropy(skorlar[:-1], hedef) #Yeni puanları cevap anahtarıyla (hedef) karşılaştırıp yeni ceza puanını hesaplıyoruz. kayip2 adını verdik ki eski kayip ile karışmasın. skorlar[:-1] son satırı atar, çünkü son harfin cevabı yok.
print(kayip2) #Yeni ceza puanını yazdırır.


#Güncellemeyi 100 kere tekrarla (eğitim döngüsü)
for adim in range(100):
    kartlar = tablo(girdi)
    skorlar = cikis(kartlar)
    kayip = nn.functional.cross_entropy(skorlar[:-1], hedef)
    optimizer.zero_grad()
    kayip.backward()
    optimizer.step()
    if adim % 10 == 0:
        print(adim, kayip.item())


with torch.no_grad(): #eğitilmiş modele “her harften sonra ne gelir?” diye sorup cevabını almak.
    tahmin = cikis(tablo(girdi)).argmax(dim=1)
print(tahmin) #Tahmin edilen numaraları yazdırır.
print("".join(numara_harf[i.item()] for i in tahmin)) #Tahmin edilen numaraları numara_harf sözlüğüyle harfe çevirip yan yana birleştirir ve yazdırır.