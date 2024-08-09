Select *
From ProjectPortfolio..['Covid Deaths$']
order by 3,4

Select *
From ProjectPortfolio..['Covid Vaccinations$']
order by 3,4

Select Location, date, total_cases, new_cases, total_deaths
FROM ProjectPortfolio..['Covid Deaths$']
order by 1,2


--Total Cases vs. Total Deaths
Select Location, date, total_cases, total_deaths, (total_deaths/NULLIF(total_cases, 0))*100 AS DeathPercentage
FROM ProjectPortfolio..['Covid Deaths$']
WHERE location like '%states%'
order by 1,2

--Total Cases vs. Population
Select Location, date, total_cases, population, (total_cases/NULLIF(population, 0))*100 AS CasePercentage
FROM ProjectPortfolio..['Covid Deaths$']
WHERE location like '%states%'
order by 1,2

--Countries with highest infection rate compared to Population
Select Location, population, MAX(total_cases) as HighestInfectionCount, MAX((total_cases/population))*100 as PercentPopulationInfected
FROM ProjectPortfolio..['Covid Deaths$']
--WHERE location like '%states%'
Group by Location, population
order by PercentPopulationInfected desc

-- Countries with Highest Death Count
Select Location, MAX(total_deaths) as TotalDeathCount
FROM ProjectPortfolio..['Covid Deaths$']
Where continent is not null
Group by Location
order by TotalDeathCount desc

--Continent with Highest Death Count
Select location, MAX(cast(total_deaths as int)) as TotalDeathCount
FROM ProjectPortfolio..['Covid Deaths$']
Where continent is null
Group by location
order by TotalDeathCount desc

--Global Death Rate of Covid Cases to Date
Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, 
SUM(cast(new_deaths as int))/SUM(NULLIF(new_cases, 0))*100 AS DeathPercentage
FROM ProjectPortfolio..['Covid Deaths$']
Where continent is not null
order by 1,2


-- Total Population vs. Vaccinations
With PopvsVac(Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(cast(vac.new_vaccinations as bigint)) OVER (Partition by dea.location order by dea.location,
 dea.date) as RollingPeopleVaccinated 
From ProjectPortfolio..['Covid Deaths$'] dea
Join ProjectPortfolio..['Covid Vaccinations$'] vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null

--order by 2,3
)
SELECT *, (RollingPeopleVaccinated/Population)*100
FROM PopvsVac

--TEMP TABLE
 
DROP Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_Vaccinations numeric,
RollingPeopleVaccinated numeric
)

Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(cast(vac.new_vaccinations as bigint)) OVER (Partition by dea.location order by dea.location,
 dea.date) as RollingPeopleVaccinated 
From ProjectPortfolio..['Covid Deaths$'] dea
Join ProjectPortfolio..['Covid Vaccinations$'] vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--order by 2,3

SELECT *, (RollingPeopleVaccinated/Population)*100
FROM #PercentPopulationVaccinated

--View for later visualizations
DROP Table if exists #PercentPopulationVaccinated
 --
Create VIEW PercentPopulationVaccinated as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(CONVERT(BIGINT, vac.new_vaccinations)) OVER (Partition by dea.location order by dea.location,
 dea.date) as RollingPeopleVaccinated 
From ProjectPortfolio..['Covid Deaths$'] dea
Join ProjectPortfolio..['Covid Vaccinations$'] vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--order by 2,3
