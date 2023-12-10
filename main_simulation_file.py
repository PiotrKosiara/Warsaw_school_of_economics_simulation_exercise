import numpy as np
import math
import matplotlib.pyplot as plt


class ProductionQueue:
    def __init__(self, initial_funds, electricity_rate, material_cost, inventories_materials, production_time,
                 quantities, price):
        self.queue = np.ndarray(shape=(quantities.shape))
        self.initial_funds = initial_funds
        self.funds = initial_funds
        self.electricity_rate = electricity_rate
        self.material_cost = material_cost
        self.inventories_materials = inventories_materials
        self.inventories_materials_init = inventories_materials
        self.production_time = production_time
        self.quantities = quantities
        self.materials = np.zeros(shape=(quantities.shape[0], inventories_materials.shape[0]))
        self.ordered_mat = np.zeros(shape=(quantities.shape[0], inventories_materials.shape[0]))
        self.price = price

    def order_materials(self, quantities):
        purchase_cost = np.zeros(shape=(self.quantities.shape))
        for i in range(0, quantities.shape[0]):
            for m in range(0, 3):
                if self.quantities[i] * np.array([0.3, 0.2, 0.7])[m] > self.inventories_materials[m]:
                    purchase_cost[i] = purchase_cost[i] + (
                                self.quantities[i] * np.array([0.3, 0.2, 0.7])[m] - self.inventories_materials[m]) * \
                                       self.material_cost[m]
                    self.materials[i, m] += self.quantities[i] * np.array([0.3, 0.2, 0.7])[m]
                    self.inventories_materials[m] = 0
                    self.ordered_mat[i, m] = self.ordered_mat[i, m] + (
                                self.quantities[i] * np.array([0.3, 0.2, 0.7])[m] - self.inventories_materials[m]) * \
                                             self.material_cost[m]
                else:
                    self.materials[i, m] = self.quantities[i] * np.array([0.3, 0.2, 0.7])[m]
                    self.inventories_materials[m] = self.inventories_materials[m] - self.quantities[i] * \
                                                    np.array([0.3, 0.2, 0.7])[m]
                    self.ordered_mat[i, m] = self.ordered_mat[i, m]

            if np.sum(purchase_cost) > self.funds:
                #print("Insufficient funds to order materials")
                return
            else:
                #print('For order %a ordered: ' % i,
                      #self.quantities[i] * np.array([0.3, 0.2, 0.7]) - self.inventories_materials,
                      #' materials for purchase_cost of: ', purchase_cost[i])
                pass

        #print('Aggregate purchase_cost of new materials: ', np.sum(purchase_cost))

        #print('Materials: ', self.materials)
        self.funds -= np.sum(purchase_cost)
        #print('Remaining funds: ', self.funds)

    def electricity_cost(self, production_time, inventories_materials):
        return self.production_time * np.sum(self.electricity_rate * self.quantities)

    def produce(self, production_time):
        self.queue = self.queue + self.production_time * self.quantities
        if self.electricity_cost(production_time, self.materials) > self.funds:
            #print("Insufficient funds for production")
            return
        self.funds -= self.electricity_cost(production_time, self.materials)
        #print("Entire production time: ", self.production_time * self.quantities.size)

    def get_queue_length(self):
        return len(self.queue)

    def revenue(self, quantities, price):
        revenue_dummy = 0
        for i in range(0, self.quantities.shape[0]):
            revenue_dummy = revenue_dummy + self.quantities[i] * self.price

        return revenue_dummy

    def profit_sales(self, quantities, price, material_cost, inventories_materials, production_time):

        return quantities * price - self.electricity_cost(self.production_time * self.quantities.size,
                                                          inventories_materials)

    def profit_accounting(self, quantities, price, material_cost, inventories_materials, production_time):
        profit_acc_dummy = self.profit_sales(self.quantities, self.price, self.material_cost,
                                             self.inventories_materials, self.production_time)
        for m in range(0, 3):
            profit_acc_dummy = profit_acc_dummy + self.inventories_materials[m] * self.material_cost[m]
        profit_acc_dummy = profit_acc_dummy - 15000
        return profit_acc_dummy

    def profit_accounting_2(self, quantities, price, material_cost, inventories_materials, production_time):
        # 15000 TO PŁACE
        return self.profit_sales(self.quantities, self.price, self.material_cost, self.inventories_materials, self.production_time) - 15000

    def remaining_inventories(self, ):
        return self.inventories_materials


# SYMULACJA WYTWÓRSTWA DLA WIELU KOLEJEK ZAMÓWIEŃ (KILKA OKRESÓW) - TRZEBA DOPISAĆ SZEREGI ZAMÓWIEŃ (quantities)!!!
queue = ProductionQueue(initial_funds=10000 * 70, electricity_rate=0.01, material_cost=np.array([10.0, 2.0, 5.0]),
                        inventories_materials=np.array([3000.0, 2000.0, 7000.0]), production_time=1.0,
                        quantities=np.array([15000.0, 1000.0, 12000.0, 7000.0]), price=15.0)
# Symulacja produkcji
queue.order_materials(np.array([15000.0, 1000.0, 12000.0, 7000.0]))
queue.produce(1.0)
print("Sales profit: ", queue.profit_sales(quantities=np.array([15000.0, 1000.0, 12000.0, 7000.0]), price=15.0,
                                           material_cost=np.array([10.0, 2.0, 5.0]),
                                           inventories_materials=np.array([3000.0, 2000.0, 7000.0]),
                                           production_time=1.0))

# PROSZĘ SPRAWDZIĆ (uruchomić queue.remaining_inventories()) - PO URUCHOMIENIU POWYŻSZYCH CZTERECH LINIJEK POLECEŃ,
# ZAPASY SĄ RÓWNE 0! TO PAŃSTWU POKAZUJE, JAK DZIAŁA .self - ODWOŁUJE SIĘ DO BIEŻĄCEJ INSTANCJI KLASY,
# TJ. W TYM PRZYPADKU - DO BIEŻĄCEJ WARTOSCI inventories_materials, KTÓRA JUŻ ZOSTAŁA ZMIENIONA
# (Z TEJ, KTÓRĄ PODALISMY JAKO ARGUMENT) POPRZEZ DZIAŁANIE FUNKCJI queue.order_materials
queue.remaining_inventories()

T = 60
# ZAŁOŻENIE - WSPÓŁCZYNNIK Z OSZACOWAŃ EKONOMETRYCZNYCH
demand_max = 28000
arr_demand_prob = np.array([])
arr_rand_draw = np.array([])
arr_demand_forecast = np.array([])
arr_rand_draw_real = np.array([])
arr_demand_prob_real = np.array([])
arr_demand = np.array([])
arr_initial_funds = np.zeros(shape=T + 1)
arr_initial_funds[0] = 10000 * 70
arr_initial_funds_2 = np.zeros(shape=T + 1)
arr_initial_funds_2[0] = 10000 * 70
profit = []
min_profit = []
max_profit = []
profit_2 = []
min_profit_2 = []
max_profit_2 = []

arr2_inventories_materials = np.zeros(shape=(T + 1, 3))
arr2_inventories_materials[0, :] = np.array([3000.0, 2000.0, 7000.0])
arr2_inventories_materials_2 = np.zeros(shape=(T + 1, 3))
arr2_inventories_materials_2[0, :] = np.array([3000.0, 2000.0, 7000.0])
for t in range(0, 60):
    # PRZEWIDYWANIE POPYTU - W TYM PRZYKŁADZIE NIE PEŁNI ONO ŻADNEJ ROLI, ALE GDYBY PRZEDSIĘBIORSTWO CHCIAŁO BRAĆ DŁUG LUB PLANOWAĆ INWESTYCJE,
    # WÓWCZAS TO BY SIĘ PRZYDAŁO. MOGĄ PAŃSTWO PORÓWNAĆ WZROKOWO RÓŻNICE POMIĘDZY PRZEWIDYWANYM A RZECZYWISTYM POPYTEM LUB POLICZYĆ MACIERZ KOWARIANCJI
    # TYCH DWÓCH WARTOŚCI - np.cov(arr_demand_forecast,arr_demand)
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    # RZECZYWISTY POPYT - NOWE LOSOWANIE ZE ZNANEGO ROZKŁADU:
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    # ZAMÓWIENIE W DANYM OKRESIE = RZECZYWISTY POPYT. NASTĘPNIE KORZYSTAMY Z NASZYCH OBIEKTÓW KOLEJKOWYCH, TJ. ProductionQueue, DO WYKONANIA
    # AUTOMATYCZNIE NIEZBĘDNYCH DO PRODUKCJI, SPRZEDAŻY I RACHUNKOWOŚCI DZIAŁAŃ
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.01,
                            material_cost=np.array([10.0, 2.0, 5.0]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                            quantities=arr_quantities, price=15.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=1.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

    arr_quantities_2 = np.array([demand])
    queue_2 = ProductionQueue(initial_funds=arr_initial_funds_2[t], electricity_rate=0.01,
                            material_cost=np.array([10.0, 2.0, 5.0]),
                            inventories_materials=arr2_inventories_materials_2[t, :], production_time=1.0,
                            quantities=arr_quantities_2, price=15.0)
    queue_2.order_materials(arr_quantities_2)
    queue_2.produce(production_time=1.0)
    arr2_inventories_materials_2[t + 1, :] = queue_2.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds_2[t + 1] = queue_2.profit_accounting_2(quantities=arr_quantities_2, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials_2[t, :],
                                                       production_time=1.0)
    profit_2.append(queue_2.profit_accounting_2(quantities=arr_quantities_2, price=15.0,
                                          material_cost=np.array([10.0, 2.0, 5.0]),
                                          inventories_materials=arr2_inventories_materials_2[t, :],
                                          production_time=1.0))
    min_profit_2.append(min(profit))
    max_profit_2.append(max(profit))


queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.01,
                        material_cost=np.array([12.0, 2.0, 5.0]),
                        inventories_materials=np.array([3000.0, 2000.0, 7000.0]), production_time=1.0,
                        quantities=np.array([15000.0, 1000.0, 12000.0, 7000.0]), price=15.0)


"""
DLA CHĘTNYCH - WIELE SZEREGÓW CZASOWYCH, W KAŻDYM OKRESIE LOSOWANIE 
(TO TAK, JAKBYSMY 100 RAZY NARAZ WYLOSOWALI SZEREG CZASOWY DŁUGOSCI 60), ZAKŁADAJĄC, ŻE ZAKŁÓCENIA SĄ NIEZALEŻNE

N = 100
T = 60
arr2_demand_prob = np.zeros(shape = (N,T))
arr2_rand_draw = np.zeros(shape = (N,T))
arr2_demand_forecast = np.zeros(shape = (N,T))
POZOSTAŁE TABLICE TAK SAMO

for n in range(0,N):
    for t in range(0,T):
        rand_draw = np.random.weibull(5)
        prob = 1/(1 + math.exp(-2*(1 -rand_draw)))
        arr2_demand_prob[n,t] = prob
        arr2_rand_draw[n,t] = rand_draw
        arr2_demand_forecast[n,t] = demand_max*arr2_demand_prob[n,t]
        POZOSTAŁE TABLICE JAK WYŻEJ, Z WYJĄTKIEM TEGO DODATKOWEGO WYMIARU n

MOŻNA TEŻ STWORZYĆ PROSTE STATYSTYKI DLA KAŻDEGO t, NP. SREDNIE, I ZŁOŻYĆ Z NICH KOLEJNY SZEREG (TABLICĘ JEDNOWYMIAROWĄ)

"""


OKRESY = np.arange(T)

print("macierz kowariancji")
print(np.cov(arr_demand_forecast,arr_demand))

fig, ax = plt.subplots()
ax.plot(OKRESY, arr_demand_forecast, label='Popyt przewidywany w czasie')
plt.title("Przewidywany popyt", fontdict=None, loc='center', pad=None)
fig.savefig(
    r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\przewidywany_popyt_w_czasie.png",
    bbox_inches='tight')
ax.legend()
plt.show()


fig, ax = plt.subplots()
ax.plot(OKRESY, arr_initial_funds[0:T], label='Fundusze w czasie')
plt.title("Fundusze", fontdict=None, loc='center', pad=None)
fig.savefig(
    r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\initial_funds_w_czasie.png",
    bbox_inches='tight')
ax.legend()
plt.show()

#Wykres faktycznego popytu w czasie
fig, ax = plt.subplots()
ax.plot(OKRESY, arr_demand[0:T], label='Popyt faktyczny w czasie')
fig.savefig(r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\rzeczywisty_popyt_w_czasie.png",bbox_inches='tight')
plt.title("Rzeczywisty popyt", fontdict=None, loc='center', pad=None)
ax.legend()
plt.show()

#Wykres ilości produktów
fig, ax = plt.subplots()
ax.plot(OKRESY, arr_demand[0:T], label='Ilość sprzedanych produktów w czasie')
fig.savefig(r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\ilosc_sprzedanych_produktów.png",bbox_inches='tight')
plt.title("Sprzedane produkty", fontdict=None, loc='center', pad=None)
ax.legend()
plt.show()



#Wykres zysku w czasie
fig, ax = plt.subplots()
ax.plot(OKRESY, profit[0:T], label='Zysk w czasie')
ax.plot(OKRESY, min_profit[0:T], linestyle='dashed', label='Minimalny zysk')
ax.plot(OKRESY, max_profit[0:T], linestyle='dashed', label='Maksymalny zysk')
fig.savefig(r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\zysk_w_czasie.png",bbox_inches='tight')
plt.title("Zysk", fontdict=None, loc='center', pad=None)
ax.legend()
plt.show()

#Wykres zysku w czasie accounting 2
fig, ax = plt.subplots()
ax.plot(OKRESY, profit_2[0:T], label='Zysk w czasie dla accounting_2')
ax.plot(OKRESY, min_profit_2[0:T], linestyle='dashed', label='Minimalny zysk')
ax.plot(OKRESY, max_profit_2[0:T], linestyle='dashed', label='Maksymalny zysk')
fig.savefig(r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\zysk_w_czasie_accounting_2.png",bbox_inches='tight')
plt.title("Zysk accounting_2", fontdict=None, loc='center', pad=None)
ax.legend()
plt.show()


########################################################################################################################
# Szok popytowy Zmiana ceny sprzedawanego produktu z 15.0 do 1.0
queue.remaining_inventories()

T = 60
# ZAŁOŻENIE - WSPÓŁCZYNNIK Z OSZACOWAŃ EKONOMETRYCZNYCH
demand_max = 28000
arr_demand_prob = np.array([])
arr_rand_draw = np.array([])
arr_demand_forecast = np.array([])
arr_rand_draw_real = np.array([])
arr_demand_prob_real = np.array([])
arr_demand = np.array([])
arr_initial_funds = np.zeros(shape=T + 1)
arr_initial_funds[0] = 10000 * 70
profit = []
arr2_inventories_materials = np.zeros(shape=(T + 1, 3))
arr2_inventories_materials[0, :] = np.array([3000.0, 2000.0, 7000.0])
min_profit = []
max_profit = []

for t in range(0, 30):
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.01,
                            material_cost=np.array([10.0, 2.0, 5.0]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                            quantities=arr_quantities, price=15.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=1.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

for t in range(30, 60):
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.01,
                            material_cost=np.array([10.0, 2.0, 5.0]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                            quantities=arr_quantities, price=7.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=1.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=7.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=7.0,
                                          material_cost=np.array([10.0, 2.0, 5.0]),
                                          inventories_materials=arr2_inventories_materials[t, :],
                                          production_time=1.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

fig, ax = plt.subplots()
ax.plot(OKRESY, profit[0:T])
plt.title("Zysk - szok w wyniku zmiany ceny produktu z 15 do 7", fontdict=None, loc='center', pad=None)
ax.plot(OKRESY, min_profit[0:T], linestyle='dashed', label='Minimalny zysk')
ax.plot(OKRESY, max_profit[0:T], linestyle='dashed', label='Maksymalny zysk')
fig.savefig(
    r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\initial_funds_w_czasie_szok.png",
    bbox_inches='tight')
ax.legend()
plt.show()
#########################################################################################################################

########################################################################################################################
# Szok w okresie 30 zepsuła się maszyna i czas produkcji zwiększył się z 1 okresu do 50 okresów
queue.remaining_inventories()

T = 60
# ZAŁOŻENIE - WSPÓŁCZYNNIK Z OSZACOWAŃ EKONOMETRYCZNYCH
demand_max = 28000
arr_demand_prob = np.array([])
arr_rand_draw = np.array([])
arr_demand_forecast = np.array([])
arr_rand_draw_real = np.array([])
arr_demand_prob_real = np.array([])
arr_demand = np.array([])
arr_initial_funds = np.zeros(shape=T + 1)
arr_initial_funds[0] = 10000 * 70
profit = []
arr2_inventories_materials = np.zeros(shape=(T + 1, 3))
arr2_inventories_materials[0, :] = np.array([3000.0, 2000.0, 7000.0])
min_profit = []
max_profit = []

for t in range(0, 30):
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.01,
                            material_cost=np.array([10.0, 2.0, 5.0]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                            quantities=arr_quantities, price=15.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=1.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

for t in range(30, 60):
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.1,
                            material_cost=np.array([10.0, 2.0, 5.0]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=50.0,
                            quantities=arr_quantities, price=15.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=50.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([10.0, 2.0, 5.0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=50.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                          material_cost=np.array([10.0, 2.0, 5.0]),
                                          inventories_materials=arr2_inventories_materials[t, :],
                                          production_time=50.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

fig, ax = plt.subplots()
ax.plot(OKRESY, profit[0:T])
plt.title("Zysk - szok w wyniku zepsucia maszyny czas produkcji z 1 do 50", fontdict=None, loc='center', pad=None)
fig.savefig(
    r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\szok_maszyna.png",
    bbox_inches='tight')
ax.plot(OKRESY, min_profit[0:T], linestyle='dashed', label='Minimalny zysk')
ax.plot(OKRESY, max_profit[0:T], linestyle='dashed', label='Maksymalny zysk')
ax.legend()
plt.show()
#########################################################################################################################

########################################################################################################################
# Szok w okresie 30 państwo wprowadziło subsydia (ceny półproduktów za darmo)
queue.remaining_inventories()

T = 60
# ZAŁOŻENIE - WSPÓŁCZYNNIK Z OSZACOWAŃ EKONOMETRYCZNYCH
demand_max = 28000
arr_demand_prob = np.array([])
arr_rand_draw = np.array([])
arr_demand_forecast = np.array([])
arr_rand_draw_real = np.array([])
arr_demand_prob_real = np.array([])
arr_demand = np.array([])
arr_initial_funds = np.zeros(shape=T + 1)
arr_initial_funds[0] = 10000 * 70
profit = []
arr2_inventories_materials = np.zeros(shape=(T + 1, 3))
arr2_inventories_materials[0, :] = np.array([0, 0, 0])
min_profit = []
max_profit = []

for t in range(0, 30):
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.1,
                            material_cost=np.array([30, 30, 30]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                            quantities=arr_quantities, price=15.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=1.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([30, 30, 30]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([30, 30, 30]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

for t in range(30, 60):
    rand_draw = np.random.weibull(5)
    prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
    arr_demand_prob = np.append(arr_demand_prob, prob)
    arr_rand_draw = np.append(arr_rand_draw, rand_draw)
    demand_forecast = int(demand_max * arr_demand_prob[t])
    arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
    rand_draw_real = np.random.weibull(5)
    prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
    arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
    demand = int(demand_max * arr_demand_prob_real[t])
    arr_demand = np.append(arr_demand, demand)
    arr_quantities = np.array([demand])
    queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.1,
                            material_cost=np.array([0, 0, 0]),
                            inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                            quantities=arr_quantities, price=15.0)
    queue.order_materials(arr_quantities)
    queue.produce(production_time=1.0)
    arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
    # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
    arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                       material_cost=np.array([0, 0, 0]),
                                                       inventories_materials=arr2_inventories_materials[t, :],
                                                       production_time=1.0)
    profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                          material_cost=np.array([0, 0, 0]),
                                          inventories_materials=arr2_inventories_materials[t, :],
                                          production_time=1.0))
    min_profit.append(min(profit))
    max_profit.append(max(profit))

fig, ax = plt.subplots()
ax.plot(OKRESY, profit[0:T])
plt.title("Zysk - szok ceny półproduktów za darmo", fontdict=None, loc='center', pad=None)
ax.plot(OKRESY, min_profit[0:T], linestyle='dashed', label='Minimalny zysk')
ax.plot(OKRESY, max_profit[0:T], linestyle='dashed', label='Maksymalny zysk')
fig.savefig(
    r"C:\Users\Piotrek\PycharmProjects\swak_lab_2_v2\szok_ceny_produktów.png",
    bbox_inches='tight')
ax.legend()
plt.show()
#########################################################################################################################

#########################################################################################################################
#Histogram dla 100 symulacji

counter = 0
avg_lst = []
mean_lst = []
total_quantieties = []

while counter < 1000:
    queue.remaining_inventories()

    T = 60
    # ZAŁOŻENIE - WSPÓŁCZYNNIK Z OSZACOWAŃ EKONOMETRYCZNYCH
    demand_max = 28000
    arr_demand_prob = np.array([])
    arr_rand_draw = np.array([])
    arr_demand_forecast = np.array([])
    arr_rand_draw_real = np.array([])
    arr_demand_prob_real = np.array([])
    arr_demand = np.array([])
    arr_initial_funds = np.zeros(shape=T + 1)
    arr_initial_funds[0] = 10000 * 70
    profit = []
    min_profit = []
    max_profit = []
    sum_quantieties = 0

    arr2_inventories_materials = np.zeros(shape=(T + 1, 3))
    arr2_inventories_materials[0, :] = np.array([3000.0, 2000.0, 7000.0])
    for t in range(0, 60):
        # PRZEWIDYWANIE POPYTU - W TYM PRZYKŁADZIE NIE PEŁNI ONO ŻADNEJ ROLI, ALE GDYBY PRZEDSIĘBIORSTWO CHCIAŁO BRAĆ DŁUG LUB PLANOWAĆ INWESTYCJE,
        # WÓWCZAS TO BY SIĘ PRZYDAŁO. MOGĄ PAŃSTWO PORÓWNAĆ WZROKOWO RÓŻNICE POMIĘDZY PRZEWIDYWANYM A RZECZYWISTYM POPYTEM LUB POLICZYĆ MACIERZ KOWARIANCJI
        # TYCH DWÓCH WARTOŚCI - np.cov(arr_demand_forecast,arr_demand)
        rand_draw = np.random.weibull(5)
        prob = 1 / (1 + math.exp(-2 * (1 - rand_draw)))
        arr_demand_prob = np.append(arr_demand_prob, prob)
        arr_rand_draw = np.append(arr_rand_draw, rand_draw)
        demand_forecast = int(demand_max * arr_demand_prob[t])
        arr_demand_forecast = np.append(arr_demand_forecast, demand_forecast)
        # RZECZYWISTY POPYT - NOWE LOSOWANIE ZE ZNANEGO ROZKŁADU:
        rand_draw_real = np.random.weibull(5)
        prob2 = 1 / (1 + math.exp(-2 * (1 - rand_draw_real)))
        arr_demand_prob_real = np.append(arr_demand_prob_real, prob2)
        demand = int(demand_max * arr_demand_prob_real[t])
        arr_demand = np.append(arr_demand, demand)
        # ZAMÓWIENIE W DANYM OKRESIE = RZECZYWISTY POPYT. NASTĘPNIE KORZYSTAMY Z NASZYCH OBIEKTÓW KOLEJKOWYCH, TJ. ProductionQueue, DO WYKONANIA
        # AUTOMATYCZNIE NIEZBĘDNYCH DO PRODUKCJI, SPRZEDAŻY I RACHUNKOWOŚCI DZIAŁAŃ
        arr_quantities = np.array([demand])
        queue = ProductionQueue(initial_funds=arr_initial_funds[t], electricity_rate=0.01,
                                material_cost=np.array([10.0, 2.0, 5.0]),
                                inventories_materials=arr2_inventories_materials[t, :], production_time=1.0,
                                quantities=arr_quantities, price=15.0)
        queue.order_materials(arr_quantities)
        queue.produce(production_time=1.0)
        arr2_inventories_materials[t + 1, :] = queue.remaining_inventories()
        # MOGĄ PAŃSTWO TEŻ ZROBIĆ TO SAMO Z .profit_accounting
        arr_initial_funds[t + 1] = queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                                           material_cost=np.array([10.0, 2.0, 5.0]),
                                                           inventories_materials=arr2_inventories_materials[t, :],
                                                           production_time=1.0)
        profit.append(queue.profit_accounting(quantities=arr_quantities, price=15.0,
                                              material_cost=np.array([10.0, 2.0, 5.0]),
                                              inventories_materials=arr2_inventories_materials[t, :],
                                              production_time=1.0))
        min_profit.append(min(profit))
        max_profit.append(max(profit))
        sum_quantieties = np.sum(arr_quantities)

    avg_lst.append(np.average(profit))
    mean_lst.append(np.median(profit))
    total_quantieties.append(sum_quantieties)
    counter = counter + 1
print("dane:")
print(min(avg_lst))
print(max(avg_lst))
print(min(mean_lst))
print(max(mean_lst))
plt.hist(avg_lst, bins=20, edgecolor='black')
plt.xlabel('Wartości')
plt.ylabel('Liczba wystąpień')
plt.title('Histogram średniej zysku')
plt.show()

plt.hist(mean_lst, bins=20, edgecolor='black')
plt.xlabel('Wartości')
plt.ylabel('Liczba wystąpień')
plt.title('Histogram mediany zysku')
plt.show()

plt.hist(total_quantieties, bins=20, edgecolor='black')
plt.xlabel('Wartości')
plt.ylabel('Liczba wystąpień')
plt.title('Histogram całkowitej ilości sprzedanych produktów przez okres T=60 dla 1000 symulacji')
plt.show()

