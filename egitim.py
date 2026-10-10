import torch #Model hesaplarını yapacak kütüphane
import torch.nn as nn #İçindeki model parçaları kutusunu (nn = neural network) alır ve kısaca nn diye çağıracağımızı söyler

metin = "merhaba" #modelin çalışacağı kelime
harfler = sorted(set(metin)) #kelimedeki harflerin alfabetik listesi
harf_numara = {h: i for i, h in enumerate(harfler)} #Her harfe sırayla bir numara veriyoruz alfabetik olarak sıraladıktan sonra 
numara_harf = {i: h for h, i in harf_numara.items()} #Burda da tam tersini yapıyoruz numaraları harfe çeviriyoruz model sonunda numara üretecek biz onu tekrar harfe çevirmek için buna ihtiyaç duyacaz

sayilar = [harf_numara[h] for h in metin] #metindeki her harfin numarasını sözlükten bulup listeye diziyoruz
print(harfler) #Harf listesini ekrana yazdırıyoruz
print(harf_numara) #Hangi harfin hangi numarayı aldığını görmek için sözlüğü ekrana yazdırıyoruz
print(sayilar) #Kelimenin numara halini ekrana yazdırıyoruz

tablo = nn.Embedding(len(harfler), 2) #bir tablo yaratıyoruz 
print(tablo.weight)
print(tablo.weight.numel())