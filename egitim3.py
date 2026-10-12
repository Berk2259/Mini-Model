import torch 
import torch.nn as nn

metin = "bugun hava cok guzel. yarin hava daha da guzel olacak. dun hava cok soguktu. bugun hava cok guzel ama yarin hava daha da guzel olacak. dun hava cok soguktu ama bugun hava cok guzel." ##Modelin öğreneceği cümle.
harfler = sorted(set(metin)) #Cümledeki farklı karakterlerin alfabetik listesi. Boşluk ve nokta da sayılıyor, toplam 19.
harf_numara = {h: i for i, h in enumerate(harfler)} #Harften numaraya sözlük.
numara_harf = {i: h for h, i in harf_numara.items()} #Numaradan harfe sözlük (tersi).
sayilar = [harf_numara[h] for h in metin] #Cümlenin tamamı numara listesine çevrilmiş hâli.
 
baglam = 8 #Model her seferinde kaç harfe bakacak.
girdi = torch.stack([torch.tensor(sayilar[i:i+baglam]) for i in range(len(sayilar) - baglam)]) #Cümlenin üzerinde 8 harflik bir pencere kaydırıyoruz. İlk pencere bugun ha, ikincisi ugun hav, üçüncüsü gun hava... Her pencere bir soru. torch.stack bütün pencereleri üst üste dizip tek bir tablo yapar: 46 satır (soru), 8 sütun (harf).
hedef = torch.tensor(sayilar[baglam:]) #Her pencereden sonra gelen doğru harf.

# Kontrol amaçlı: tabloların boyutunu [46, 8] ve [46] olarak yazar, ilk soruyu (bugun ha) ve cevabını (v) harfle gösterir.
print(girdi.shape)
print(hedef.shape)
print("".join(numara_harf[i.item()] for i in girdi[0]))
print(numara_harf[hedef[0].item()])

tablo = nn.Embedding(len(harfler), 2) #: Kart tablosu. 19 harfin her birine 2 sayılık kart. 19 × 2 = 38 parametre.
cikis = nn.Linear(baglam * 2, len(harfler)) # Puanlama katmanı. 16 sayı girer (8 kart × 2 sayı), 19 puan çıkar (her harf için bir puan). 16 × 19 + 19 = 323 parametre. Toplam model: 361 parametre.

kartlar = tablo(girdi).reshape(len(girdi), -1) #46 sorunun her biri için 8 kart çeker.Her sorunun 8 kartını yan yana tek satıra dizer, 16 sayı olur. -1 “kalanı sen hesapla” demek.
skorlar = cikis(kartlar) #Her soru için 19 harfin puanı (tahmin).
kayip = nn.functional.cross_entropy(skorlar, hedef) #Tahminleri doğru cevaplarla karşılaştırıp ceza puanı hesaplar.

#Boyutları ([46, 16], [46, 19]) ve eğitimden önceki ceza puanını yazdırır.
print(kartlar.shape)
print(skorlar.shape)
print(kayip)

#Parametreleri güncelleyecek aracı hazırlar: 361 parametrenin hepsini, her seferinde 0,5 güçle günceller.
optimizer = torch.optim.SGD(list(tablo.parameters()) + list(cikis.parameters()), lr=0.5)

#Eğitim Döngüsü
for adim in range(100):
    kartlar = tablo(girdi).reshape(len(girdi), -1)
    skorlar = cikis(kartlar)
    kayip = nn.functional.cross_entropy(skorlar, hedef)
    optimizer.zero_grad()
    kayip.backward()
    optimizer.step()
    if adim % 10 == 0:
        print(adim, kayip.item())

# Sadece soruyor öğnremiyor, 46 Soruya modelin cevabın: her satırda en yüksek puanlı harf
with torch.no_grad():
    kartlar = tablo(girdi).reshape(len(girdi), -1)
    tahmin = cikis(kartlar).argmax(dim=1)

print("".join(numara_harf[i.item()] for i in tahmin)) #Modelin cevaplarını harfe çevirip yan yana yazar.
print("".join(numara_harf[i.item()] for i in hedef)) #Doğru cevapları yazar.
print((tahmin == hedef).sum().item(), "/", len(hedef))#Cevapları tek tek karşılaştırır, doğruları sayar