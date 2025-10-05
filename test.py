import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering

# 1️⃣ Şehir özellik verisini oluştur
data = {
    'City': ['Ankara', 'İstanbul', 'İzmir', 'Bursa', 'Antalya'],
    'Population': [5.0, 15.0, 4.5, 3.0, 2.5],      # milyon
    'AverageIncome': [80, 100, 85, 70, 75],        # bin TL
    'Temperature': [12, 15, 18, 14, 22]            # ortalama sıcaklık
}

df = pd.DataFrame(data)
print("📊 Orijinal veri:")
print(df)

# 2️⃣ Şehirlerin sayısal özelliklerini çıkar
X = df[['Population', 'AverageIncome', 'Temperature']].values


# 3️⃣ Öklid uzaklık matrisi hesapla
#Bu satır, her bir şehir çifti arasındaki “uzaklığı” (farkı) bulur.
#Yani, şehirlerin nüfus, gelir ve sıcaklık değerlerini birer nokta gibi düşünürsek, bu kod iki şehir arasındaki düzlemdeki mesafeyi (Öklidyen mesafe) hesaplar.
#Sonuçta oluşan matris, her şehrin diğer şehirlerle ne kadar “benzer” veya “farklı” olduğunu sayısal olarak gösterir.
dist_matrix = euclidean_distances(X, X) 


# 4️⃣ Benzerlik (ağırlık) matrisi oluştur (Gauss kernel)
#Bu kod, şehirler arasındaki benzerlikleri hesaplamak için kullanılır.
#Burada Gauss (veya RBF) çekirdeği uygulanır:

#sigma değeri, benzerliğin ne kadar "yayılacağını" belirleyen bir ölçek parametresidir.
#similarity_matrix ise, şehirler arasındaki mesafeleri (dist_matrix) alıp, bunları 0 ile 1 arasında bir benzerlik skoruna dönüştürür.
#Mesafe küçükse benzerlik 1’e yakın, mesafe büyükse benzerlik 0’a yakın olur.
#Kısacası:
#Bu satır, şehirler birbirine ne kadar yakınsa o kadar yüksek, uzaksa düşük bir benzerlik puanı verir.
#Bu puanlar, graf ve kümeleme işlemlerinde kullanılır.
sigma = 10  # ölçek parametresi
similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))



#euclidean_distances(X, X) → Şehirler arasındaki mesafeyi (farkı) verir. Mesafe küçükse şehirler birbirine benzer, mesafe büyükse farklıdır.
#similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2)) → Bu mesafeleri benzerlik skoruna çevirir. Sonuç 0 ile 1 arasındadır. Mesafe küçükse benzerlik 1’e yakın olur, mesafe büyükse 0’a yaklaşır.




print("\n🔗 Benzerlik (ağırlık) matrisi:")
print(pd.DataFrame(similarity_matrix, index=df['City'], columns=df['City']))

# 5️⃣ Graf oluştur (ağırlıklar benzerliklerden)
G = nx.Graph()
for i in range(len(df)):
    for j in range(i+1, len(df)):
        weight = similarity_matrix[i, j]
        if weight > 0.2:  # zayıf bağlantıları dışlayabiliriz (isteğe bağlı)
            G.add_edge(df['City'][i], df['City'][j], weight=weight)

# 6️⃣ Grafı çiz
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)

plt.figure(figsize=(7,5))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):f"{d['weight']:.2f}" for u,v,d in edges})
plt.title("Şehirler Arası Benzerlik Grafı (ağırlıklar veriden hesaplandı)")
plt.show()

# 7️⃣ Spektral kümeleme uygula
n_clusters = 2
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
labels = sc.fit_predict(similarity_matrix)

df['Cluster'] = labels
print("\n🧠 Kümeleme Sonucu:")
print(df[['City', 'Cluster']])

# 8️⃣ Küme sonuçlarını graf üzerinde renklendir
node_order = list(G.nodes())
color_map = ['orange' if labels[df.index[df['City'] == node][0]] == 0 else 'lightgreen' for node in node_order]
plt.figure(figsize=(7,5))
nx.draw(G, pos, with_labels=True, nodelist=node_order, node_color=color_map, node_size=2000, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):f"{d['weight']:.2f}" for u,v,d in edges})
plt.title("Spektral Kümeleme Sonucu (ağırlıklar otomatik çıkarıldı)")
plt.show()
